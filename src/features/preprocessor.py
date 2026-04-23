"""
Etapas 1 y 2 del pipeline:
  1. Detección de rostro y alineación geométrica con MTCNN.
  2. Preprocesamiento: resize → CLAHE → sharpening suave.

Diseño:
  - FaceDetector            : detecta y alinea el rostro (transf. afín sobre 5 landmarks).
  - ImagePreprocessor       : aplica las transformaciones fotométricas (CLAHE + sharpening).
  - FacePreprocessingPipeline: devuelve DOS imágenes del mismo recorte alineado:
      · face_raw        → sin filtros fotométricos, para el LivenessDetector.
      · face_processed  → con CLAHE + sharpening, para el FaceEmbedder.

    Por qué separar liveness del preprocesamiento:
      CLAHE y sharpening modifican el histograma y los bordes de alta frecuencia
      que el DenseNet201 usa como señal de spoof (patrones Moiré de impresoras,
      reflexiones especulares de pantallas, granularidad de papel fotográfico).
      Aplicar estos filtros antes del liveness puede suprimir exactamente esas
      señales, reduciendo la sensibilidad del detector a ataques de presentación.
      El embedder, en cambio, se beneficia de la normalización fotométrica para
      ser robusto a variaciones de iluminación inter-sesión.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from loguru import logger

import torch


# Estructuras de datos 

@dataclass
class DetectionResult:
    """Resultado de detección de MTCNN para un único rostro."""
    bbox: np.ndarray        # [x1, y1, x2, y2] float32
    confidence: float
    landmarks: np.ndarray   # shape (5, 2): [ojo_izq, ojo_der, nariz, boca_izq, boca_der]
    aligned_face: np.ndarray  # imagen BGR recortada y alineada, shape (H, W, 3)


# Etapa 1: Detección y Alineación 

# Coordenadas de referencia para los 5 landmarks en la imagen de salida 224x224.
# Derivadas del estándar ArcFace (112x112) escaladas x2.
_REFERENCE_LANDMARKS_224 = np.array([
    [73.55,  90.68],   # ojo izquierdo
    [150.45, 90.68],   # ojo derecho
    [112.0,  130.0],   # nariz
    [83.5,   162.0],   # comisura boca izquierda
    [140.5,  162.0],   # comisura boca derecha
], dtype=np.float32)


class FaceDetector:
    """
    Detecta y alinea el rostro dominante en una imagen usando MTCNN.

    La alineación geométrica (transformación afín mínima con 5 puntos) corrige
    la pose en roll/pitch/yaw leve y estandariza la posición de los ojos,
    condición necesaria para que ArcFace opere en su margen angular óptimo.
    """

    def __init__(
        self,
        device: str = "cpu",
        min_face_size: int = 40,
        thresholds: list[float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        output_size: int = 224,
    ):
        self.output_size = output_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # keep_all=False → sólo el rostro con mayor probabilidad (autenticación 1:1)
        self.mtcnn = MTCNN(
            image_size=output_size,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            keep_all=False,
            device=self.device,
        )
        logger.debug(f"FaceDetector inicializado en dispositivo: {self.device}")

    def detect(self, image_bgr: np.ndarray) -> Optional[DetectionResult]:
        """
        Detecta y alinea el rostro en `image_bgr`.

        Returns:
            DetectionResult si se detecta un rostro con confianza suficiente,
            None en caso contrario.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # MTCNN devuelve (boxes, probs, landmarks) cuando return_prob=True
        boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)

        if boxes is None or probs is None or probs[0] is None:
            logger.debug("MTCNN: no se detectó ningún rostro.")
            return None

        if probs[0] < self.mtcnn.thresholds[-1]:
            logger.debug(f"MTCNN: confianza insuficiente ({probs[0]:.3f}).")
            return None

        bbox = boxes[0].astype(np.float32)
        lm = landmarks[0].astype(np.float32)  # shape (5, 2)

        aligned = self._align(image_rgb, lm)

        return DetectionResult(
            bbox=bbox,
            confidence=float(probs[0]),
            landmarks=lm,
            aligned_face=cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
        )

    def _align(self, image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Transformación afín parcial (similitud) que mapea los 5 landmarks
        detectados sobre las coordenadas de referencia en 224x224.

        Se usa estimateAffinePartial2D en lugar de la afín completa para
        preservar el aspecto (sin distorsión).
        """
        M, _ = cv2.estimateAffinePartial2D(
            landmarks, _REFERENCE_LANDMARKS_224, method=cv2.LMEDS
        )
        if M is None:
            # Fallback: recorte simple si la estimación falla (oclusión severa)
            logger.warning("Alineación afín fallida, usando recorte directo.")
            h, w = image_rgb.shape[:2]
            aligned = cv2.resize(image_rgb, (self.output_size, self.output_size))
            return aligned

        aligned = cv2.warpAffine(
            image_rgb, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned


# Etapa 2: Preprocesamiento Fotométrico 

class ImagePreprocessor:
    """
    Aplica CLAHE y sharpening al rostro alineado para maximizar la robustez
    ante variaciones de iluminación.

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
      - Opera en el canal L del espacio LAB para no afectar el color.
      - clip_limit=2.0: compromiso entre contraste local y artefactos de
        halos; valores >4 introducen ruido visible.
      - tile_grid_size=(8,8): bloques de ~28x28 px en una imagen 224x224,
        suficiente granularidad para sombras de un solo lado del rostro.

    Sharpening (kernel unsharp mask suave):
      - kernel_strength=0.3: realza bordes de cejas, iris y contornos faciales
        sin amplificar el ruido de sensor; crítico para imágenes comprimidas.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        kernel_strength: float = 0.3,
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )
        self.kernel_strength = kernel_strength

    def process(self, face_bgr: np.ndarray) -> np.ndarray:
        """Aplica CLAHE + sharpening. Input y output: BGR uint8."""
        # CLAHE en canal L 
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Sharpening (unsharp mask) 
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(
            enhanced, 1.0 + self.kernel_strength,
            blurred, -self.kernel_strength,
            0,
        )
        return sharpened


# Pipeline completo (Etapas 1+2) 

class FacePreprocessingPipeline:
    """
    Combina FaceDetector + ImagePreprocessor en una sola llamada.

    Devuelve TRES valores para permitir que cada etapa posterior reciba
    exactamente la imagen que necesita:
      - face_raw       : recorte alineado sin filtros fotométricos.
                         → usado por LivenessDetector (preserva señales de spoof).
      - face_processed : recorte alineado con CLAHE + sharpening aplicados.
                         → usado por FaceEmbedder (robusto a iluminación).
      - detection_result: bbox, landmarks y confianza de MTCNN.

    Si no se detecta rostro, lanza ValueError para abortar el pipeline en cascada.
    """

    def __init__(self, cfg: dict):
        det_cfg = cfg.get("detection", {})
        pre_cfg = cfg.get("preprocessing", {})
        clahe_cfg = pre_cfg.get("clahe", {})
        sharp_cfg = pre_cfg.get("sharpening", {})

        self.detector = FaceDetector(
            device=cfg.get("pipeline", {}).get("device", "cpu"),
            min_face_size=det_cfg.get("min_face_size", 40),
            thresholds=det_cfg.get("thresholds", [0.6, 0.7, 0.7]),
            factor=det_cfg.get("factor", 0.709),
            output_size=pre_cfg.get("output_size", [224, 224])[0],
        )
        self.preprocessor = ImagePreprocessor(
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
            kernel_strength=sharp_cfg.get("kernel_strength", 0.3),
        )

    def run(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, DetectionResult]:
        """
        Ejecuta detección + alineación y genera las dos variantes de la imagen.

        Returns:
            (face_raw, face_processed, detection_result)
              - face_raw       : BGR uint8, alineado, sin filtros fotométricos.
              - face_processed : BGR uint8, alineado, con CLAHE + sharpening.
              - detection_result: metadatos MTCNN (bbox, landmarks, confidence).
        Raises:
            ValueError si no se detecta un rostro válido.
        """
        result = self.detector.detect(image_bgr)
        if result is None:
            raise ValueError("No se detectó ningún rostro en la imagen.")

        face_raw = result.aligned_face                        # sin filtros
        face_processed = self.preprocessor.process(face_raw)  # CLAHE + sharpen

        return face_raw, face_processed, result
