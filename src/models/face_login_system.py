"""
Pipeline en Cascada completo para autenticación biométrica facial.

Flujo de verificación (login):
  1. FacePreprocessingPipeline  -> detección MTCNN + alineación + CLAHE/sharpen
  2. LivenessDetector           -> DenseNet201 antispoofing (gate duro: score ≥ 0.95)
  3. FaceEmbedder               -> ArcFace R100, embedding 512-D
  4. EncryptedEmbeddingStore    -> recuperar embedding de referencia descifrado
  5. Similitud coseno           -> accept si sim ≥ threshold (0.68)
  6. AccessController           -> bloqueo tras max_failed_attempts

Flujo de enrolamiento (register):
  1-3 igual que verificación
  4. Almacenar embedding cifrado en la DB
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from loguru import logger

from src.features.preprocessor import FacePreprocessingPipeline
from src.models.liveness_detector import LivenessDetector
from src.models.face_embedder import FaceEmbedder
from src.utils.security import EncryptedEmbeddingStore, AccessController


# Códigos de resultado

class AuthStatus(Enum):
    SUCCESS          = auto()   # Autenticación exitosa
    FACE_NOT_FOUND   = auto()   # MTCNN no detectó rostro
    LIVENESS_FAILED  = auto()   # Ataque de presentación detectado
    USER_NOT_FOUND   = auto()   # user_id no registrado en la DB
    SCORE_TOO_LOW    = auto()   # Similitud coseno < umbral
    ACCOUNT_LOCKED   = auto()   # Demasiados intentos fallidos


@dataclass
class AuthResult:
    status: AuthStatus
    user_id: str
    liveness_score: Optional[float] = None
    similarity_score: Optional[float] = None
    message: str = ""

    @property
    def granted(self) -> bool:
        return self.status == AuthStatus.SUCCESS


# Sistema principal

class FaceLoginSystem:
    """
    Sistema de login biométrico facial end-to-end.

    Ejemplo mínimo:
        system = FaceLoginSystem.from_config("config/config.yaml", passphrase="secret")
        system.register("alice", cv2.imread("alice.jpg"))
        result = system.authenticate("alice", cv2.imread("probe.jpg"))
        if result.granted:
            print("Acceso concedido")
    """

    def __init__(
        self,
        cfg: dict,
        passphrase: str,
        db_path: str = "models/embeddings.pkl.enc",
    ):
        self.cfg = cfg
        ver_cfg = cfg.get("verification", {})
        live_cfg = cfg.get("liveness", {})
        sec_cfg = cfg.get("security", {})

        # Seleccionar umbral según el backend activo
        emb_backend = cfg.get("embedder", {}).get("primary", "arcface")
        if emb_backend == "arcface":
            self.sim_threshold = float(ver_cfg.get("threshold", 0.68))
        else:
            self.sim_threshold = float(ver_cfg.get("threshold_facenet", 0.60))

        # Inicializar componentes del pipeline
        self.preproc_pipeline = FacePreprocessingPipeline(cfg)
        self.liveness_detector = LivenessDetector(
            model_path=live_cfg.get("model_path", "models/liveness_densenet201.pth"),
            threshold=float(live_cfg.get("threshold", 0.95)),
            device=cfg.get("pipeline", {}).get("device", "cpu"),
        )
        self.embedder = FaceEmbedder(cfg)
        self.db = EncryptedEmbeddingStore(
            db_path=Path(db_path),
            passphrase=passphrase,
        )
        self.access_ctrl = AccessController(
            max_attempts=int(sec_cfg.get("max_failed_attempts", 5)),
            lockout_seconds=int(sec_cfg.get("lockout_seconds", 300)),
        )

        logger.info(
            f"FaceLoginSystem inicializado | backend={emb_backend} | "
            f"sim_threshold={self.sim_threshold}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "config/config.yaml",
        passphrase: str = "",
        db_path: str = "models/embeddings.pkl.enc",
    ) -> "FaceLoginSystem":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, passphrase=passphrase, db_path=db_path)

    # Enrolamiento

    def register(self, user_id: str, image_bgr: np.ndarray) -> bool:
        """
        Registra un nuevo usuario extrayendo y almacenando su embedding.

        Returns:
            True si el registro fue exitoso.
        Raises:
            ValueError si no se detecta rostro o el liveness falla.
        """
        logger.info(f"Enrolamiento de usuario '{user_id}'...")

        # Etapa 1: Detección + alineación -> face_raw (sin filtros) + face_processed
        face_raw, face_processed, det = self.preproc_pipeline.run(image_bgr)

        # Etapa 3: Liveness sobre imagen SIN preprocesar (preserva señales de spoof)
        live_score, is_live = self.liveness_detector.predict(face_raw)
        if not is_live:
            raise ValueError(
                f"Enrolamiento rechazado para '{user_id}': "
                "liveness fallido (rostro no real)."
            )

        # Etapa 4: Embedding sobre imagen CON CLAHE + sharpening
        emb_result = self.embedder.embed(face_processed)

        # Guardar cifrado
        self.db.store(user_id, emb_result.embedding)
        self.db.save()

        logger.info(
            f"Usuario '{user_id}' registrado | "
            f"liveness={live_score:.4f} | backend={emb_result.backend}"
        )
        return True

    def register_from_multiple(
        self, user_id: str, images: list[np.ndarray]
    ) -> bool:
        """
        Registra con múltiples imágenes promediando los embeddings para mayor
        robustez (ej. diferentes ángulos, iluminaciones).
        """
        embeddings = []
        for i, img in enumerate(images):
            try:
                face_raw, face_processed, _ = self.preproc_pipeline.run(img)
                live_score, is_live = self.liveness_detector.predict(face_raw)
                if not is_live:
                    logger.warning(
                        f"Imagen {i} de '{user_id}' descartada (liveness={live_score:.3f})"
                    )
                    continue
                emb = self.embedder.embed(face_processed).embedding
                embeddings.append(emb)
            except ValueError as e:
                logger.warning(f"Imagen {i} de '{user_id}' omitida: {e}")

        if not embeddings:
            raise ValueError(
                f"No se pudo procesar ninguna imagen válida para '{user_id}'."
            )

        # Promediar y renormalizar (centroide en la hiperesfera)
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        self.db.store(user_id, mean_emb)
        self.db.save()
        logger.info(
            f"Usuario '{user_id}' registrado con {len(embeddings)} muestras."
        )
        return True

    # Verificación (login)

    def authenticate(self, user_id: str, image_bgr: np.ndarray) -> AuthResult:
        """
        Ejecuta el pipeline en cascada completo para verificar una identidad.

        Cada etapa puede abortar el flujo retornando un AuthResult con el
        código de fallo correspondiente. Solo cuando todas las etapas pasan
        se retorna AuthStatus.SUCCESS.
        """

        # Control de acceso (pre-gate)
        if self.access_ctrl.is_locked(user_id):
            return AuthResult(
                status=AuthStatus.ACCOUNT_LOCKED,
                user_id=user_id,
                message="Cuenta bloqueada temporalmente por exceso de intentos.",
            )

        live_score: Optional[float] = None
        sim_score: Optional[float] = None

        # Etapa 1: Detección + alineación (genera raw y processed)
        try:
            face_raw, face_processed, det = self.preproc_pipeline.run(image_bgr)
        except ValueError as e:
            logger.warning(f"[{user_id}] Etapa 1 fallida: {e}")
            return AuthResult(
                status=AuthStatus.FACE_NOT_FOUND,
                user_id=user_id,
                message=str(e),
            )

        # Etapa 3: Liveness sobre imagen SIN filtros (preserva artefactos)
        live_score, is_live = self.liveness_detector.predict(face_raw)
        if not is_live:
            logger.warning(
                f"[{user_id}] LIVENESS FAILED: score={live_score:.4f}"
            )
            # No se registra como intento fallido de contraseña; es un ataque
            return AuthResult(
                status=AuthStatus.LIVENESS_FAILED,
                user_id=user_id,
                liveness_score=live_score,
                message="Ataque de presentación detectado.",
            )

        # Etapa 4: Extracción de embedding sobre imagen CON CLAHE + sharpen 
        probe_result = self.embedder.embed(face_processed)

        # Etapa 4b: Recuperar embedding de referencia 
        ref_embedding = self.db.retrieve(user_id)
        if ref_embedding is None:
            logger.warning(f"[{user_id}] Usuario no encontrado en la DB.")
            return AuthResult(
                status=AuthStatus.USER_NOT_FOUND,
                user_id=user_id,
                liveness_score=live_score,
                message=f"Usuario '{user_id}' no está registrado.",
            )

        # Etapa 5: Verificación por similitud coseno 
        sim_score = self.embedder.cosine_similarity(
            probe_result.embedding, ref_embedding
        )
        logger.info(
            f"[{user_id}] sim={sim_score:.4f} | "
            f"threshold={self.sim_threshold} | live={live_score:.4f}"
        )

        if sim_score < self.sim_threshold:
            self.access_ctrl.record_failure(user_id)
            remaining = self.access_ctrl.remaining_attempts(user_id)
            return AuthResult(
                status=AuthStatus.SCORE_TOO_LOW,
                user_id=user_id,
                liveness_score=live_score,
                similarity_score=sim_score,
                message=f"Verificación fallida. Intentos restantes: {remaining}.",
            )

        # Éxito 
        self.access_ctrl.record_success(user_id)
        logger.info(f"[{user_id}] Autenticación EXITOSA.")
        return AuthResult(
            status=AuthStatus.SUCCESS,
            user_id=user_id,
            liveness_score=live_score,
            similarity_score=sim_score,
            message="Acceso concedido.",
        )

    # Gestión de usuarios 

    def remove_user(self, user_id: str) -> bool:
        removed = self.db.delete(user_id)
        if removed:
            self.db.save()
        return removed

    def list_users(self) -> list[str]:
        return self.db.list_users()

    def is_registered(self, user_id: str) -> bool:
        return user_id in self.db
