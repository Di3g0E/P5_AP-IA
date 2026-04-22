"""
Módulo de evaluación biométrica.

Métricas implementadas:
  - EER (Equal Error Rate)     : punto donde FAR == FRR.
  - ROC-AUC                    : área bajo la curva ROC.
  - TAR@FAR=k                  : True Accept Rate a una tasa de False Accept fija.
  - Curva DET (Detection Error Tradeoff): visualización estándar en ISO/IEC 19795.

Benchmark:
  - BiometricBenchmark: evalúa y compara ArcFace vs SFace sobre un dataset
    de pares etiquetados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from scipy.interpolate import interp1d
from scipy.special import erfinv
from sklearn.metrics import roc_auc_score, roc_curve


# ── Funciones de métricas ─────────────────────────────────────────────────────

def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Calcula el Equal Error Rate (EER) y el umbral correspondiente.

    Args:
        y_true : array de etiquetas binarias (1=genuino, 0=impostor).
        scores : array de puntuaciones de similitud (coseno).

    Returns:
        (eer, threshold): EER ∈ [0, 1] y el umbral en el punto EER.

    Nota: el EER es el punto donde FAR = FRR. Se interpola linealmente entre
    los dos puntos más cercanos de la curva ROC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr  # FRR = 1 - TPR

    # Índice del cruce FAR ≈ FRR
    abs_diff = np.abs(fpr - fnr)
    min_idx = int(np.argmin(abs_diff))

    # Interpolación lineal para mayor precisión
    if min_idx == 0 or min_idx >= len(fpr) - 1:
        eer = float(fpr[min_idx])
        eer_threshold = float(thresholds[min_idx])
    else:
        f_far = interp1d(
            [fpr[min_idx - 1], fpr[min_idx]],
            [thresholds[min_idx - 1], thresholds[min_idx]],
        )
        f_frr = interp1d(
            [fnr[min_idx - 1], fnr[min_idx]],
            [thresholds[min_idx - 1], thresholds[min_idx]],
        )
        # Umbral en el cruce
        eer_threshold = float((f_far(fpr[min_idx]) + f_frr(fnr[min_idx])) / 2)
        eer = float((fpr[min_idx] + fnr[min_idx]) / 2)

    return eer, eer_threshold


def compute_tar_at_far(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_far: float = 1e-3,
) -> float:
    """
    Calcula TAR (True Accept Rate) al nivel de FAR especificado.

    En biometría el TAR@FAR=0.001 es el indicador estándar de comparación
    entre sistemas (ISO/IEC 19795-1).
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    # Interpolar TPR en el punto FAR objetivo
    interp = interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
    tar = float(np.clip(interp(target_far), 0.0, 1.0))
    return tar


def compute_full_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Calcula el conjunto completo de métricas para un umbral dado.

    Returns:
        dict con keys: eer, eer_threshold, roc_auc, tar_at_far_1e3,
                       tar_at_far_1e4, accuracy, precision, recall, f1.
    """
    eer, eer_thr = compute_eer(y_true, scores)
    roc_auc = float(roc_auc_score(y_true, scores))
    tar_1e3 = compute_tar_at_far(y_true, scores, target_far=1e-3)
    tar_1e4 = compute_tar_at_far(y_true, scores, target_far=1e-4)

    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    far       = fp / max(fp + tn, 1)
    frr       = fn / max(fn + tp, 1)

    metrics = {
        "eer": eer,
        "eer_threshold": eer_thr,
        "roc_auc": roc_auc,
        "tar_at_far_1e3": tar_1e3,
        "tar_at_far_1e4": tar_1e4,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far_at_threshold": far,
        "frr_at_threshold": frr,
    }

    logger.info(
        f"Métricas | EER={eer:.4f} @ thr={eer_thr:.4f} | "
        f"AUC={roc_auc:.4f} | TAR@FAR=1e-3={tar_1e3:.4f}"
    )
    return metrics


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "Sistema",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Genera y opcionalmente guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatorio")
    ax.set_xlabel("Tasa de Falsa Aceptación (FAR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdadera Aceptación (TAR)", fontsize=12)
    ax.set_title("Curva ROC — Verificación Facial", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
        logger.info(f"Curva ROC guardada en {output_path}")

    return fig


def plot_det_curve(
    systems: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Curva DET (Detection Error Tradeoff) para comparar múltiples sistemas.

    Args:
        systems: {nombre: (y_true, scores)} para cada sistema a comparar.
    """

    def _probit(p: np.ndarray) -> np.ndarray:
        """Transformación probit (escala normal) para curvas DET."""
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.sqrt(2) * erfinv(2 * p - 1)

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (y_true, scores) in systems.items():
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
        fnr = 1 - tpr
        ax.plot(
            _probit(fpr), _probit(fnr),
            linewidth=2, label=name,
        )

    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    tick_labels = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "40%"]
    probit_ticks = _probit(np.array(ticks))

    ax.set_xticks(probit_ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(probit_ticks)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlim(_probit(0.0005), _probit(0.5))
    ax.set_ylim(_probit(0.0005), _probit(0.5))
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("FRR (%)", fontsize=12)
    ax.set_title("Curva DET — Comparativa de Sistemas", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
        logger.info(f"Curva DET guardada en {output_path}")

    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Distribución de scores genuinos vs impostores con umbral marcado."""
    genuine_scores  = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(genuine_scores,  ax=ax, fill=True, label="Genuinos",  color="steelblue", alpha=0.6)
    sns.kdeplot(impostor_scores, ax=ax, fill=True, label="Impostores", color="tomato",    alpha=0.6)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Umbral={threshold:.3f}")
    ax.set_xlabel("Similitud del Coseno", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de Scores — Genuinos vs. Impostores", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)

    return fig


# ── Benchmark comparativo ─────────────────────────────────────────────────────

class BiometricBenchmark:
    """
    Evalúa y compara múltiples backends de embedding (ArcFace vs SFace)
    sobre un dataset de pares verificación etiquetado.

    Uso:
        bench = BiometricBenchmark(pair_dataset, system)
        report = bench.run(thresholds={"arcface": 0.68, "sface": 0.60})
        bench.save_report(report, "doc/evaluation")
    """

    def __init__(self, pair_dataset, login_system):
        """
        Args:
            pair_dataset : VerificationPairDataset
            login_system : FaceLoginSystem (con acceso al preprocesador y embedder)
        """
        self.pair_dataset = pair_dataset
        self.system = login_system

    def _score_pairs(self, backend: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae scores y etiquetas para todos los pares del dataset con el
        backend especificado.
        """
        scores_list: List[float] = []
        labels_list: List[int] = []

        original_backend = self.system.embedder.primary_backend
        self.system.embedder.primary_backend = backend

        for img1, img2, label in self.pair_dataset:
            try:
                face1, _ = self.system.preproc_pipeline.run(img1)
                face2, _ = self.system.preproc_pipeline.run(img2)
                emb1 = self.system.embedder.embed(face1).embedding
                emb2 = self.system.embedder.embed(face2).embedding
                sim  = self.system.embedder.cosine_similarity(emb1, emb2)
                scores_list.append(sim)
                labels_list.append(int(label))
            except Exception as e:
                logger.warning(f"Par omitido durante benchmark: {e}")

        self.system.embedder.primary_backend = original_backend
        return np.array(labels_list), np.array(scores_list)

    def run(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        output_dir: str = "doc/evaluation",
    ) -> Dict[str, Dict]:
        """
        Ejecuta el benchmark completo y genera todos los gráficos.

        Returns:
            {backend: metrics_dict}
        """
        if thresholds is None:
            thresholds = {"arcface": 0.68, "facenet": 0.60}

        report: Dict[str, Dict] = {}
        det_data: Dict[str, Tuple] = {}

        for backend, thr in thresholds.items():
            logger.info(f"Evaluando backend: {backend}...")
            y_true, scores = self._score_pairs(backend)

            if len(y_true) == 0:
                logger.error(f"No se obtuvieron scores para {backend}.")
                continue

            metrics = compute_full_metrics(y_true, scores, threshold=thr)
            report[backend] = metrics
            det_data[backend] = (y_true, scores)

            # Curva ROC individual
            plot_roc_curve(
                y_true, scores,
                label=backend.upper(),
                output_path=f"{output_dir}/roc_{backend}.png",
            )
            # Distribución de scores
            plot_score_distribution(
                y_true, scores,
                threshold=thr,
                output_path=f"{output_dir}/score_dist_{backend}.png",
            )

        # DET comparativa
        if len(det_data) > 1:
            plot_det_curve(
                det_data,
                output_path=f"{output_dir}/det_comparison.png",
            )

        return report

    def save_report(
        self, report: Dict[str, Dict], output_dir: str = "doc/evaluation"
    ) -> None:
        """Guarda el informe de métricas en formato CSV."""
        import pandas as pd
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(report).T
        csv_path = Path(output_dir) / "benchmark_report.csv"
        df.to_csv(csv_path)
        logger.info(f"Informe de benchmark guardado en {csv_path}")
        print(df.to_string())
