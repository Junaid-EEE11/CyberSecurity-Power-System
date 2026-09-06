"""Anomaly score trajectory and threshold visualization."""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_scores_timeline(
    scores: np.ndarray,
    is_attack: np.ndarray,
    threshold: float,
    timesteps: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot anomaly score sequence with calibrated threshold and attack bands (Figure 5)."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    ax.plot(timesteps, scores, color="#1f77b4", label="Composite Anomaly Score", linewidth=1.2)
    ax.axhline(threshold, color="#d62728", linestyle="--", label=f"Calibrated Threshold (tau={threshold:.3f})", linewidth=1.5)

    # Highlight ground truth attack intervals
    in_attack = False
    start_t = 0
    for i in range(len(is_attack)):
        if is_attack[i] == 1 and not in_attack:
            in_attack = True
            start_t = timesteps[i]
        elif is_attack[i] == 0 and in_attack:
            in_attack = False
            ax.axvspan(start_t, timesteps[i - 1], color="#ff7f0e", alpha=0.25, label="Ground Truth Attack" if start_t == timesteps[np.where(is_attack == 1)[0][0]] else "")

    if in_attack:
        ax.axvspan(start_t, timesteps[-1], color="#ff7f0e", alpha=0.25)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Standardized Anomaly Score", fontsize=11)
    ax.set_title("Anomaly Score Progression Across Normal & Attacked Windows", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
