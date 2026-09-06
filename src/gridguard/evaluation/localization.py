"""Node-level cyberattack localization evaluation metrics."""

from typing import Dict, List, Optional
import numpy as np


def compute_localization_metrics(
    y_node_true: np.ndarray,
    node_scores: np.ndarray,
    is_attack_sample: np.ndarray,
    node_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate attack localization metrics across compromised timesteps.

    Args:
        y_node_true: Binary ground truth matrix [N_samples, N_nodes] (1 if compromised, 0 otherwise).
        node_scores: Predicted continuous node anomaly scores [N_samples, N_nodes].
        is_attack_sample: Global binary indicator [N_samples] (1 if sample contains an attack).
        node_threshold: Optional threshold for node-level binary prediction.

    Returns:
        Dictionary of Top-1 hit rate, Top-3 hit rate, Top-5 hit rate, Node Precision, Recall, and F1.
    """
    attack_indices = np.where(is_attack_sample == 1)[0]
    if len(attack_indices) == 0:
        return {
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "node_precision": 0.0,
            "node_recall": 0.0,
            "node_f1": 0.0,
        }

    top1_hits = []
    top3_hits = []
    top5_hits = []

    precisions = []
    recalls = []
    f1s = []

    for idx in attack_indices:
        gt_nodes = np.where(y_node_true[idx] == 1)[0]
        if len(gt_nodes) == 0:
            continue

        scores_i = node_scores[idx]
        sorted_nodes = np.argsort(-scores_i)

        # Top-k Hits
        top1_hits.append(1.0 if sorted_nodes[0] in gt_nodes else 0.0)
        top3_hits.append(1.0 if len(set(sorted_nodes[:3]) & set(gt_nodes)) > 0 else 0.0)
        top5_hits.append(1.0 if len(set(sorted_nodes[:5]) & set(gt_nodes)) > 0 else 0.0)

        # Threshold-based node classification
        thresh = node_threshold if node_threshold is not None else np.percentile(scores_i, 95)
        pred_nodes = np.where(scores_i > thresh)[0]

        tp = len(set(pred_nodes) & set(gt_nodes))
        fp = len(set(pred_nodes) - set(gt_nodes))
        fn = len(set(gt_nodes) - set(pred_nodes))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "top1_hit": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "top3_hit": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "top5_hit": float(np.mean(top5_hits)) if top5_hits else 0.0,
        "node_precision": float(np.mean(precisions)) if precisions else 0.0,
        "node_recall": float(np.mean(recalls)) if recalls else 0.0,
        "node_f1": float(np.mean(f1s)) if f1s else 0.0,
    }
