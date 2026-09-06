"""Detection performance metrics calculation (AUROC, AUPRC, F1, Recall@FAR)."""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute complete detection performance metrics.

    Args:
        y_true: Ground truth binary attack labels [0 or 1].
        scores: Continuous anomaly scores.
        threshold: Optional threshold for discrete metrics. If None, median score is used.

    Returns:
        Dictionary containing AUROC, AUPRC, Precision, Recall, F1, FPR, and FAR.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    # Calculate threshold-independent rank metrics
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception:
        auprc = float(np.mean(y_true))

    if threshold is None:
        threshold = float(np.percentile(scores[y_true == 0], 95)) if (y_true == 0).any() else float(np.median(scores))

    y_pred = (scores > threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    far = fpr  # False Alarm Rate on normal operational samples

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "far": far,
        "threshold": float(threshold),
    }


def compute_recall_at_far(y_true: np.ndarray, scores: np.ndarray, target_far: float = 0.05) -> float:
    """Compute attack recall when threshold is set to achieve exact empirical target FAR on normal samples."""
    normal_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]

    if len(normal_scores) == 0 or len(attack_scores) == 0:
        return 0.0

    threshold = np.percentile(normal_scores, (1.0 - target_far) * 100.0)
    recall = float(np.mean(attack_scores > threshold))
    return recall
