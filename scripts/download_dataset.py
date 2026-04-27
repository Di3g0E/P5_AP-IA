"""
Descarga el dataset Kaggle de liveness usando las credenciales del .env.

Variables esperadas en .env:
    KAGGLE_USERNAME=...
    KAGGLE_KEY=...

Uso:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --slug trainingdatapro/real-vs-fake-anti-spoofing-video-classification
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

DEFAULT_SLUG = "trainingdatapro/real-vs-fake-anti-spoofing-video-classification"


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga un dataset de Kaggle vía kagglehub.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="owner/dataset")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_dotenv(args.env_file)

    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        logger.error(
            "Faltan KAGGLE_USERNAME y/o KAGGLE_KEY en el entorno. "
            "Añádelos a .env (mira .env.example)."
        )
        sys.exit(1)

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub no está instalado. Ejecuta: pip install kagglehub")
        sys.exit(1)

    logger.info(f"Descargando dataset '{args.slug}' como {os.environ['KAGGLE_USERNAME']}...")
    path = kagglehub.dataset_download(args.slug)
    logger.success(f"Dataset disponible en: {path}")

    p = Path(path)
    for split in ("train", "test"):
        split_dir = p / split
        if split_dir.is_dir():
            sub = sorted(c.name for c in split_dir.iterdir() if c.is_dir())
            logger.info(f"  {split}/: {sub}")
        else:
            logger.warning(f"  {split}/ no encontrado")

    print(f"\n[OK] DATA_PATH={path}")


if __name__ == "__main__":
    main()
