"""
Etapa 4: Extracción de Embeddings Faciales.

  - ArcFaceEmbedder  : DeepFace + modelo ArcFace (ResNet-100, 512-D).
                       Descarga los pesos la primera vez y los cachea en ~/.deepface/.
  - FaceNetEmbedder  : facenet-pytorch InceptionResnetV1 (512-D).
                       Pesos preentrenados en VGGFace2. Alternativa ligera.
  - FaceEmbedder     : clase unificada que selecciona el backend por config.

Por qué ArcFace (vía DeepFace) como primario:
  - Margen angular aditivo m=0.5 fuerza mayor separabilidad inter-clase en el
    espacio hiperesférico (norma L2 = 1). En LFW: TAR@FAR=1e-3 > 99.8%.
  - DeepFace expone el mismo modelo ArcFace R100 entrenado en MS-Celeb-1M.

Por qué FaceNet (facenet-pytorch) como alternativa:
  - 100 % PyTorch puro → wheels en todas las plataformas sin compilación.
  - InceptionResnetV1 preentrenado en VGGFace2 produce embeddings 512-D.
  - Umbral de coseno ajustado a 0.60 (vs 0.68 de ArcFace) por su menor
    discriminabilidad en el margen angular.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


# Resultado de embedding

@dataclass
class EmbeddingResult:
    embedding: np.ndarray    # float32, shape (512,), norma L2 = 1
    backend: str             # "arcface" | "facenet"
    inference_ms: float      # latencia de inferencia


# ArcFace vía DeepFace

class ArcFaceEmbedder:
    """
    Extrae embeddings ArcFace usando la librería DeepFace.

    DeepFace descarga los pesos del modelo ArcFace la primera vez que se
    llama a represent() y los almacena en ~/.deepface/weights/. No requiere
    ninguna compilación; todos sus backends son Python puro o wheels oficiales.

    Se usa enforce_detection=False porque la detección ya fue realizada en
    la Etapa 1 (MTCNN): se le pasa directamente el recorte alineado.
    """

    MODEL_NAME = "ArcFace"   # 512-D, entrenado en MS-Celeb-1M

    def __init__(self):
        # Importación diferida para no bloquear el arranque si deepface no
        # está instalado (el error se lanza solo al intentar usar este backend)
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            # Precarga del modelo para evitar latencia en el primer login
            logger.info("Precargando modelo ArcFace (DeepFace)...")
            self._deepface.build_model(self.MODEL_NAME)
            logger.info("ArcFaceEmbedder (DeepFace) inicializado.")
        except ImportError:
            raise ImportError(
                "Instala DeepFace: pip install deepface"
            )

    def embed(self, aligned_face_bgr: np.ndarray) -> EmbeddingResult:
        """
        Args:
            aligned_face_bgr: recorte de rostro alineado BGR uint8, 224x224.
        Returns:
            EmbeddingResult con embedding L2-normalizado de 512 dimensiones.
        """
        t0 = time.perf_counter()

        # DeepFace.represent espera RGB; además necesita 152x152 para ArcFace
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (152, 152))

        result = self._deepface.represent(
            img_path=face_resized,
            model_name=self.MODEL_NAME,
            enforce_detection=False,  # detección ya hecha en Etapa 1
            detector_backend="skip",
            align=False,              # alineación ya hecha en Etapa 1
        )

        embedding = np.array(result[0]["embedding"], dtype=np.float32)

        # Garantizar norma L2 = 1 (DeepFace lo hace, pero añadimos robustez)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"ArcFace (DeepFace) embedding en {ms:.1f} ms")
        return EmbeddingResult(embedding=embedding, backend="arcface", inference_ms=ms)


# FaceNet vía facenet-pytorch

class FaceNetEmbedder:
    """
    Extrae embeddings usando InceptionResnetV1 de facenet-pytorch.

    facenet-pytorch es PyTorch puro: no requiere compilación en ninguna
    plataforma. Pesos preentrenados en VGGFace2 → 512 dimensiones.

    Preferido cuando:
      - DeepFace no está disponible.
      - Se necesita inferencia más rápida (InceptionResnet < DenseNet).
      - El entorno tiene restricciones de memoria (modelo más ligero).
    """

    def __init__(self, device: str = "cpu", pretrained: str = "vggface2"):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            self._device = torch.device(
                device if __import__("torch").cuda.is_available() else "cpu"
            )
            self._model = InceptionResnetV1(pretrained=pretrained).eval()
            self._model = self._model.to(self._device)
            self._torch = torch
            logger.info(
                f"FaceNetEmbedder (InceptionResnetV1/{pretrained}) "
                f"inicializado en {self._device}."
            )
        except ImportError:
            raise ImportError(
                "Instala facenet-pytorch: pip install facenet-pytorch"
            )

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> "torch.Tensor":
        """Convierte BGR uint8 → tensor float32 normalizado [-1, 1], shape (1,3,160,160)."""
        import torch
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_160 = cv2.resize(face_rgb, (160, 160)).astype(np.float32)
        # Normalización estándar FaceNet: [0,255] → [-1,1]
        face_160 = (face_160 - 127.5) / 128.0
        tensor = torch.from_numpy(face_160.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def embed(self, aligned_face_bgr: np.ndarray) -> EmbeddingResult:
        import torch
        t0 = time.perf_counter()
        tensor = self._preprocess(aligned_face_bgr).to(self._device)
        with torch.no_grad():
            embedding = self._model(tensor).squeeze().cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"FaceNet (facenet-pytorch) embedding en {ms:.1f} ms")
        return EmbeddingResult(embedding=embedding, backend="facenet", inference_ms=ms)


# Clase unificada

class FaceEmbedder:
    """
    Fachada que expone ArcFace (DeepFace) y FaceNet (facenet-pytorch) bajo
    una interfaz única y proporciona un benchmark comparativo.

    Backends disponibles:
      "arcface"  → ArcFaceEmbedder  (512-D, DeepFace, requiere deepface)
      "facenet"  → FaceNetEmbedder  (512-D, facenet-pytorch, PyTorch puro)
    """

    def __init__(self, cfg: dict, model_root: str = "models"):
        emb_cfg = cfg.get("embedder", {})
        self.primary_backend = emb_cfg.get("primary", "arcface")
        self.model_root = model_root
        self._device = cfg.get("pipeline", {}).get("device", "cpu")

        self._arcface: Optional[ArcFaceEmbedder] = None
        self._facenet: Optional[FaceNetEmbedder] = None

        if self.primary_backend == "arcface":
            self._arcface = ArcFaceEmbedder()
        else:
            self._facenet = FaceNetEmbedder(device=self._device)

    def embed(self, aligned_face_bgr: np.ndarray) -> EmbeddingResult:
        """Extrae embedding con el backend primario configurado."""
        if self.primary_backend == "arcface":
            if self._arcface is None:
                self._arcface = ArcFaceEmbedder()
            return self._arcface.embed(aligned_face_bgr)
        
        if self._facenet is None:
            self._facenet = FaceNetEmbedder(device=getattr(self, '_device', 'cpu'))
        return self._facenet.embed(aligned_face_bgr)

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Similitud del coseno entre dos embeddings L2-normalizados.
        Equivalente al producto punto cuando norm(v) = 1.
        Rango: [-1, 1]; 1 = mismo sujeto.
        """
        return float(np.dot(emb_a, emb_b))

    def benchmark(
        self,
        test_faces: List[np.ndarray],
        n_runs: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark comparativo ArcFace (DeepFace) vs FaceNet (facenet-pytorch).

        Mide latencia media, throughput y similitud coseno entre pares de
        imágenes consecutivas en `test_faces`.
        """
        results: Dict[str, Dict[str, float]] = {}

        # Cargar ambos backends para el benchmark
        if self._arcface is None:
            self._arcface = ArcFaceEmbedder()
        if self._facenet is None:
            self._facenet = FaceNetEmbedder(device=getattr(self, '_device', 'cpu'))

        for name, embedder in [("arcface", self._arcface), ("facenet", self._facenet)]:
            latencies: List[float] = []
            embeddings_list: List[np.ndarray] = []

            for face in test_faces[:n_runs]:
                res = embedder.embed(face)
                latencies.append(res.inference_ms)
                embeddings_list.append(res.embedding)

            lat_arr = np.array(latencies)
            sims = [
                self.cosine_similarity(embeddings_list[i], embeddings_list[i + 1])
                for i in range(0, len(embeddings_list) - 1, 2)
            ]

            results[name] = {
                "mean_latency_ms": float(np.mean(lat_arr)),
                "p95_latency_ms":  float(np.percentile(lat_arr, 95)),
                "throughput_fps":  float(1000.0 / np.mean(lat_arr)),
                "mean_cosine_sim": float(np.mean(sims)) if sims else 0.0,
            }

            logger.info(
                f"[Benchmark] {name}: "
                f"lat={results[name]['mean_latency_ms']:.1f} ms | "
                f"fps={results[name]['throughput_fps']:.1f} | "
                f"cos_sim={results[name]['mean_cosine_sim']:.4f}"
            )

        return results
