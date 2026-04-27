"""
Evaluación del detector de Liveness sobre la partición de test del dataset
Kaggle (`real-vs-fake-anti-spoofing-video-classification`).

Estructura esperada del dataset:
    <data_path>/
        test/
            real_video/*.mp4   -> label=1 (live)
            attack/*.mp4       -> label=0 (spoof)

Uso:
    python scripts/eval_liveness.py \
        --data-path "C:/path/to/kagglehub/.../versions/1" \
        --split test \
        --frames-per-video 3 \
        --output-dir doc/evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.liveness_detector import LivenessDetector  # noqa: E402


def extract_frames(video_path: Path, target_seconds: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    for sec in target_seconds:
        idx = int(sec * fps)
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def collect_split(
    data_path: Path, split: str, frames_per_video: int
) -> Tuple[List[np.ndarray], List[int]]:
    target_seconds = [3, 5, 7][:frames_per_video]
    frames: List[np.ndarray] = []
    labels: List[int] = []

    classes = [("real_video", 1), ("attack", 0)]
    for subdir, label in classes:
        folder = data_path / split / subdir
        if not folder.is_dir():
            raise FileNotFoundError(f"No existe la carpeta esperada: {folder}")
        videos = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp4")
        logger.info(f"[{split}/{subdir}] {len(videos)} vídeos")
        for v in videos:
            for f in extract_frames(v, target_seconds):
                frames.append(f)
                labels.append(label)

    n_live = sum(labels)
    logger.info(
        f"Frames extraídos: {len(frames)} (live={n_live}, spoof={len(labels) - n_live})"
    )
    return frames, labels


def compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thr[eer_idx])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_threshold": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "frr_at_threshold": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_samples": int(len(labels)),
        "n_live": int(labels.sum()),
        "n_spoof": int(len(labels) - labels.sum()),
        "_curves": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()},
    }


def save_sample_grid(
    frames: List[np.ndarray],
    labels: List[int],
    scores: np.ndarray,
    output_dir: Path,
    n_per_class: int = 4,
    seed: int = 42,
) -> None:
    """Guarda un mosaico con N ejemplos live y N spoof, anotados con label y score."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    live_idx = np.where(labels_arr == 1)[0]
    spoof_idx = np.where(labels_arr == 0)[0]
    if len(live_idx) == 0 or len(spoof_idx) == 0:
        logger.warning("No hay muestras de ambas clases; se omite el mosaico.")
        return

    n = min(n_per_class, len(live_idx), len(spoof_idx))
    pick_live = rng.choice(live_idx, size=n, replace=False)
    pick_spoof = rng.choice(spoof_idx, size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(pick_live):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"REAL (live)\nscore={scores[idx]:.3f}", color="green", fontsize=10)
        axes[0, col].axis("off")

    for col, idx in enumerate(pick_spoof):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(rgb)
        axes[1, col].set_title(f"FALSO (spoof)\nscore={scores[idx]:.3f}", color="red", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle("Ejemplos del conjunto de test — Liveness", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "samples_live_vs_spoof.png", dpi=150, bbox_inches="tight")
    plt.close()

    # También guarda un ejemplo individual de cada clase
    for tag, idx in (("live", int(pick_live[0])), ("spoof", int(pick_spoof[0]))):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        color = "green" if tag == "live" else "red"
        label = "REAL (live)" if tag == "live" else "FALSO (spoof)"
        plt.title(f"{label} — score={scores[idx]:.3f}", color=color)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_outputs(scores: np.ndarray, labels: np.ndarray, metrics: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fpr = np.array(metrics["_curves"]["fpr"])
    tpr = np.array(metrics["_curves"]["tpr"])

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.scatter([metrics["far_at_threshold"]], [1 - metrics["frr_at_threshold"]],
                color="red", zorder=5, label=f"thr={metrics['threshold']:.2f}")
    plt.xlabel("FAR (False Accept Rate)")
    plt.ylabel("TAR (True Accept Rate)")
    plt.title("ROC — Liveness")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_liveness.png", dpi=150)
    plt.close()

    # Histograma de scores
    plt.figure(figsize=(6, 5))
    plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="live", color="green")
    plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="spoof", color="red")
    plt.axvline(metrics["threshold"], color="black", linestyle="--",
                label=f"thr={metrics['threshold']:.2f}")
    plt.axvline(metrics["eer_threshold"], color="orange", linestyle=":",
                label=f"EER thr={metrics['eer_threshold']:.2f}")
    plt.xlabel("Liveness score")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de scores — Liveness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_dist_liveness.png", dpi=150)
    plt.close()

    # Matriz de confusión
    cm = metrics["confusion_matrix"]
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["spoof", "live"])
    plt.yticks([0, 1], ["spoof", "live"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión — Liveness")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_liveness.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del modelo de liveness sobre la partición de test.")
    parser.add_argument("--data-path", required=True, help="Raíz del dataset Kaggle (con subcarpeta test/).")
    parser.add_argument("--split", default="test", choices=["test", "train"], help="Partición a evaluar.")
    parser.add_argument("--frames-per-video", type=int, default=3)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-path", default=None, help="Override del modelo (default: el de config.yaml).")
    parser.add_argument("--threshold", type=float, default=None, help="Override del threshold (default: el de config.yaml).")
    parser.add_argument("--output-dir", default=None, help="Override del directorio de salida.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_path = args.model_path or cfg["liveness"]["model_path"]
    threshold = args.threshold if args.threshold is not None else cfg["liveness"]["threshold"]
    output_dir = Path(args.output_dir or cfg.get("evaluation", {}).get("output_dir", "doc/evaluation"))
    device = args.device or cfg.get("pipeline", {}).get("device", "cpu")

    detector = LivenessDetector(model_path=model_path, threshold=threshold, device=device)

    frames, labels = collect_split(Path(args.data_path), args.split, args.frames_per_video)
    if not frames:
        logger.error("No se han extraído frames; abortando.")
        sys.exit(1)

    logger.info("Inferencia en batch...")
    scores: List[float] = []
    BATCH = 16
    for i in range(0, len(frames), BATCH):
        batch = frames[i:i + BATCH]
        results = detector.predict_batch(batch)
        scores.extend(s for s, _ in results)

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    metrics = compute_metrics(scores_arr, labels_arr, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in metrics.items() if k != "_curves"}
    with open(output_dir / "liveness_test_report.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
    with open(output_dir / "liveness_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_outputs(scores_arr, labels_arr, metrics, output_dir)
    save_sample_grid(frames, labels, scores_arr, output_dir)

    logger.info("Resultados:")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer", "eer_threshold"):
        logger.info(f"  {k:<15}: {report[k]:.4f}")
    logger.info(f"  confusion_matrix: {report['confusion_matrix']}")
    logger.info(f"Reporte guardado en: {output_dir}")


if __name__ == "__main__":
    main()
