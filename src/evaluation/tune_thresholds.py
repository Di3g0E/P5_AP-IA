"""
Optimización de hiperparámetros del sistema biométrico.

Dos procedimientos independientes:

  1. tune_verification_threshold():
        Barre todos los valores de threshold posibles sobre los scores
        de pairs.csv (genuino vs impostor) y selecciona el óptimo por
        tres criterios (EER, F1 máximo, accuracy máxima). Útil cuando
        tenemos un dataset de pares etiquetados.

  2. tune_liveness_threshold():
        Al no disponer de ataques de presentación reales, calibra el
        threshold contra la DISTRIBUCIÓN de scores sobre rostros reales:
        el threshold se fija en el percentil-p (p.ej. p=5) para tolerar
        un 5 % de FRR operacional en usuarios legítimos.

El tuner puede además escribir los valores óptimos de vuelta al config.yaml
para que los próximos logins los usen automáticamente.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger
from sklearn.metrics import f1_score, roc_curve

from src.data.face_dataset import VerificationPairDataset, load_single_image
from src.evaluation.metrics import (
    compute_eer,
    plot_roc_curve,
    plot_score_distribution,
)


# ── Resultado de la búsqueda ─────────────────────────────────────────────────

@dataclass
class ThresholdSearchResult:
    """Resumen de la búsqueda de threshold óptimo."""
    eer:                float
    eer_threshold:      float
    best_f1:            float
    best_f1_threshold:  float
    best_acc:           float
    best_acc_threshold: float
    n_genuine:          int
    n_impostor:         int
    roc_auc:            float


# ── 1. Verificación ──────────────────────────────────────────────────────────

def _score_all_pairs(pair_dataset, system) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae (labels, scores) coseno sobre todos los pares del dataset."""
    labels, scores = [], []
    n_skipped = 0
    for i, (img1, img2, label) in enumerate(pair_dataset):
        try:
            _, face1, _ = system.preproc_pipeline.run(img1)
            _, face2, _ = system.preproc_pipeline.run(img2)
            emb1 = system.embedder.embed(face1).embedding
            emb2 = system.embedder.embed(face2).embedding
            sim = system.embedder.cosine_similarity(emb1, emb2)
            scores.append(sim)
            labels.append(int(label))
        except Exception as e:
            n_skipped += 1
            logger.warning(f"Par {i} omitido ({e})")
    if n_skipped:
        logger.warning(f"{n_skipped}/{len(pair_dataset)} pares no procesables.")
    return np.array(labels), np.array(scores)


def tune_verification_threshold(
    system,
    pair_dataset,
    search_steps: int = 1000,
    output_dir: str | Path = "doc/evaluation",
) -> ThresholdSearchResult:
    """
    Optimiza el threshold de similitud coseno sobre `pair_dataset`.

    Genera:
      - output_dir/score_dist_tuning.png   distribución de scores con umbrales marcados
      - output_dir/roc_tuning.png          curva ROC
      - output_dir/tuning_report.yaml      resumen numérico

    Returns:
        ThresholdSearchResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scoring {len(pair_dataset)} pares...")
    y_true, scores = _score_all_pairs(pair_dataset, system)

    n_genuine = int((y_true == 1).sum())
    n_impostor = int((y_true == 0).sum())
    if n_genuine < 2 or n_impostor < 2:
        raise ValueError(
            f"Se necesitan al menos 2 genuinos y 2 impostores "
            f"(hay {n_genuine}/{n_impostor})."
        )

    # EER (punto FAR==FRR) ─────────────────────────────────────────────────
    eer, eer_thr = compute_eer(y_true, scores)

    # Búsqueda de grilla para F1 y accuracy ────────────────────────────────
    candidates = np.linspace(scores.min(), scores.max(), search_steps)
    f1s = np.array([f1_score(y_true, scores >= t, zero_division=0) for t in candidates])
    accs = np.array([(((scores >= t) == y_true).mean()) for t in candidates])

    best_f1_idx = int(np.argmax(f1s))
    best_acc_idx = int(np.argmax(accs))

    # ROC-AUC ──────────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score
    roc_auc = float(roc_auc_score(y_true, scores))

    result = ThresholdSearchResult(
        eer=float(eer),
        eer_threshold=float(eer_thr),
        best_f1=float(f1s[best_f1_idx]),
        best_f1_threshold=float(candidates[best_f1_idx]),
        best_acc=float(accs[best_acc_idx]),
        best_acc_threshold=float(candidates[best_acc_idx]),
        n_genuine=n_genuine,
        n_impostor=n_impostor,
        roc_auc=roc_auc,
    )

    # ── Gráficos ────────────────────────────────────────────────────────
    plot_score_distribution(
        y_true, scores,
        threshold=result.eer_threshold,
        output_path=output_dir / "score_dist_tuning.png",
    )
    plot_roc_curve(
        y_true, scores,
        label="Verificación",
        output_path=output_dir / "roc_tuning.png",
    )
    _plot_threshold_sweep(
        candidates, f1s, accs,
        eer_thr=result.eer_threshold,
        output_path=output_dir / "threshold_sweep.png",
    )

    # ── Informe ─────────────────────────────────────────────────────────
    with open(output_dir / "tuning_report.yaml", "w") as f:
        yaml.safe_dump(asdict(result), f, sort_keys=False)

    logger.info(
        f"Tuning | EER={result.eer:.4f} @ {result.eer_threshold:.4f} | "
        f"bestF1={result.best_f1:.4f} @ {result.best_f1_threshold:.4f} | "
        f"AUC={result.roc_auc:.4f}"
    )
    return result


def _plot_threshold_sweep(
    candidates: np.ndarray,
    f1s: np.ndarray,
    accs: np.ndarray,
    eer_thr: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(candidates, f1s, label="F1", linewidth=2)
    ax.plot(candidates, accs, label="Accuracy", linewidth=2)
    ax.axvline(eer_thr, color="black", linestyle="--", alpha=0.6,
               label=f"EER thr={eer_thr:.3f}")
    ax.set_xlabel("Threshold de similitud coseno")
    ax.set_ylabel("Métrica")
    ax.set_title("Barrido de thresholds — Verificación")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ── 2. Liveness ──────────────────────────────────────────────────────────────

def tune_liveness_threshold(
    system,
    real_images_dir: str | Path,
    target_frr: float = 0.05,
    output_dir: str | Path = "doc/evaluation",
) -> Dict[str, float]:
    """
    Calibra el threshold de liveness usando la distribución de scores sobre
    rostros reales (sin ataques disponibles).

    Args:
        real_images_dir : directorio con imágenes reales del usuario.
        target_frr      : FRR operacional máximo tolerado (0.05 = 5 %).
                          El threshold se fija en el percentil target_frr*100
                          de los scores: el 5 % inferior de scores reales se
                          rechazaría erróneamente.

    Returns:
        dict con threshold recomendado y estadísticas.

    Advertencia: este método NO mide FAR contra ataques. Para validar FAR real
    hay que etiquetar datos de spoof (fotos, pantallas, máscaras).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_images_dir = Path(real_images_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in sorted(real_images_dir.rglob("*")) if p.suffix.lower() in exts]

    if len(image_paths) < 5:
        logger.warning(
            f"Sólo {len(image_paths)} imágenes reales — calibración poco fiable. "
            f"Se recomiendan ≥ 30 para un threshold estable."
        )

    scores: List[float] = []
    for p in image_paths:
        try:
            img = load_single_image(p)
            face_raw, _, _ = system.preproc_pipeline.run(img)
            score, _ = system.liveness_detector.predict(face_raw)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Imagen {p.name} omitida: {e}")

    if not scores:
        raise RuntimeError("No se pudo extraer ningún score de liveness.")

    scores_arr = np.array(scores)
    percentile_thr = float(np.percentile(scores_arr, target_frr * 100))

    # Histograma de scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_arr, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(percentile_thr, color="red", linestyle="--", linewidth=2,
               label=f"threshold p{int(target_frr*100)}={percentile_thr:.3f}")
    ax.set_xlabel("Liveness score (DenseNet201)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de scores en rostros reales ({len(scores_arr)} imgs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "liveness_calibration.png"), dpi=150)
    plt.close(fig)

    summary = {
        "recommended_threshold":    percentile_thr,
        "target_frr":               target_frr,
        "n_samples":                len(scores_arr),
        "mean_score":               float(scores_arr.mean()),
        "std_score":                float(scores_arr.std()),
        "min_score":                float(scores_arr.min()),
        "p25":                      float(np.percentile(scores_arr, 25)),
        "median":                   float(np.median(scores_arr)),
        "p75":                      float(np.percentile(scores_arr, 75)),
        "max_score":                float(scores_arr.max()),
    }

    with open(output_dir / "liveness_tuning.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    logger.info(
        f"Liveness tuning | threshold={percentile_thr:.4f} "
        f"(percentil {int(target_frr*100)} sobre {len(scores_arr)} reales)"
    )
    return summary


# ── 3. Escribir óptimos en config.yaml ───────────────────────────────────────

def update_config(
    config_path: str | Path,
    verification_threshold: Optional[float] = None,
    liveness_threshold: Optional[float] = None,
    criterion: str = "eer",
) -> None:
    """
    Actualiza in-place los thresholds en config.yaml sin perder comentarios
    que no toca.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if verification_threshold is not None:
        cfg["verification"]["threshold"] = round(float(verification_threshold), 4)
        logger.info(
            f"verification.threshold = {cfg['verification']['threshold']} "
            f"(criterio: {criterion})"
        )

    if liveness_threshold is not None:
        cfg["liveness"]["threshold"] = round(float(liveness_threshold), 4)
        logger.info(f"liveness.threshold = {cfg['liveness']['threshold']}")

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
