"""Calibration curves and empirical False-Alarm Rate visualization."""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(
    alpha_targets: List[float],
    empirical_fars: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot target significance alpha vs empirical clean False-Alarm Rate (Figure 10)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.plot([0, 0.20], [0, 0.20], "k--", label="Ideal Calibration (Target = Empirical)", linewidth=1.5)
    ax.plot(alpha_targets, empirical_fars, "ro-", markersize=7, linewidth=2, label="GridGuard Conformal Calibration")

    ax.set_xlabel(r"Nominal Target False-Alarm Rate ($\alpha$)", fontsize=11)
    ax.set_ylabel(r"Empirical Test False-Alarm Rate ($\mathrm{FAR}_{\mathrm{emp}}$)", fontsize=11)
    ax.set_title("Split-Conformal Calibration Reliability", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 0.20])
    ax.set_ylim([0.0, 0.20])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
