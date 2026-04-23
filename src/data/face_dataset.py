"""
Dataset utilities para el sistema de login biométrico.

Soporta dos modos:
  - Enrolamiento: cargar imágenes de usuario y extraer embeddings de referencia.
  - Evaluación:   cargar pares (imagen_prueba, identidad_esperada) para calcular
                  métricas de verificación (EER, ROC-AUC).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger
from torch.utils.data import Dataset


@dataclass
class FaceRecord:
    """Representa un registro en la base de datos de identidades."""
    user_id: str
    embedding: np.ndarray          # vector float32, shape (512,)
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class FaceImageDataset(Dataset):
    """
    Dataset PyTorch para cargar imágenes de rostros desde un directorio con
    estructura estándar:

        root/
          user_alice/  img_001.jpg  img_002.jpg ...
          user_bob/    img_001.jpg  ...

    Cada subdirectorio es una identidad; las imágenes son muestras de esa
    identidad. Devuelve (imagen_bgr: np.ndarray, label: str).
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, str]] = []
        self.labels: list[str] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.root_dir}")

        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            label = identity_dir.name
            for img_path in sorted(identity_dir.iterdir()):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, label))
                    if label not in self.labels:
                        self.labels.append(label)

        logger.info(
            f"Dataset cargado: {len(self.samples)} imágenes, "
            f"{len(self.labels)} identidades desde '{self.root_dir}'"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"No se pudo leer la imagen: {img_path}")
        if self.transform:
            img = self.transform(img)
        return img, label

    def iter_by_identity(self) -> Iterator[tuple[str, list[np.ndarray]]]:
        """Agrupa las imágenes por identidad (útil para enrolamiento batch)."""
        from collections import defaultdict
        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for img, label in self:
            groups[label].append(img)
        for label, imgs in groups.items():
            yield label, imgs


class VerificationPairDataset(Dataset):
    """
    Dataset de pares para evaluación de verificación 1:1.

    Formato del archivo CSV (sin cabecera):
        img1_path,img2_path,1   <- mismo sujeto (genuine)
        img1_path,img2_path,0   <- distinto sujeto (impostor)
    """

    def __init__(self, pairs_csv: str | Path, transform=None):
        self.pairs_csv = Path(pairs_csv)
        self.transform = transform
        self.pairs: list[tuple[Path, Path, int]] = []
        self._load_csv()

    def _load_csv(self) -> None:
        if not self.pairs_csv.exists():
            raise FileNotFoundError(f"CSV de pares no encontrado: {self.pairs_csv}")

        # Las rutas relativas en el CSV se resuelven respecto al directorio
        # del propio CSV (no al CWD), para que el pairs.csv sea portable.
        csv_dir = self.pairs_csv.resolve().parent

        def _resolve(p: str) -> Path:
            path = Path(p)
            if not path.is_absolute():
                path = (csv_dir / path).resolve()
            return path

        with open(self.pairs_csv) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    logger.warning(f"Línea {lineno} malformada, se omite: {line}")
                    continue
                p1, p2, label = _resolve(parts[0]), _resolve(parts[1]), int(parts[2])
                self.pairs.append((p1, p2, label))

        genuine = sum(1 for *_, l in self.pairs if l == 1)
        impostor = len(self.pairs) - genuine
        logger.info(
            f"Pares cargados: {genuine} genuinos, {impostor} impostores"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        p1, p2, label = self.pairs[idx]
        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        if img1 is None or img2 is None:
            raise IOError(f"No se pudo leer el par de imágenes: {p1}, {p2}")
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, label


def load_single_image(path: str | Path) -> np.ndarray:
    """Lee una imagen desde disco y valida que no esté corrupta."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Imagen no válida o corrupta: {path}")
    return img
