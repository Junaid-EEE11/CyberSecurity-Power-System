"""Comparative results, PR curves, robustness, localization, and ablation plots."""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_precision_recall_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparative Precision-Recall curves across all baselines (Figure 6)."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    colors = {
        "B0_Statistical_MAD": "#8c564b",
        "B1_PCA_SPE": "#7f7f7f",
        "B2_IsolationForest": "#bcbd22",
        "B3_Dense_AE": "#17becf",
        "B4_LSTM_AE": "#ff7f0e",
        "B5_PhysicsResidualOnly": "#2ca02c",
        "B6_GraphTemporal_AE": "#9467bd",
        "B7_Proposed_Physics_GTAE": "#d62728",
    }

    for model_name, data in results_dict.items():
        scores = data["scores"]
        prec, rec, _ = precision_recall_curve(y_true, scores)
        auprc = data.get("auprc", 0.0)
        c = colors.get(model_name, "#333333")
        lw = 2.5 if "Proposed" in model_name else 1.5
        ax.plot(rec, prec, label=f"{model_name} (AUPRC={auprc:.3f})", color=c, linewidth=lw)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Benchmark for FDIA Detection", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_attack_strength_sensitivity(
    strengths: List[float],
    model_auprc_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot detection AUPRC as a function of attack strength (Figure 7)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for model_name, auprc_vals in model_auprc_dict.items():
        lw = 2.5 if "Proposed" in model_name else 1.5
        ax.plot(strengths, auprc_vals, marker="o", label=model_name, linewidth=lw)

    ax.set_xlabel("Attack Perturbation Strength", fontsize=11)
    ax.set_ylabel("Detection AUPRC", fontsize=11)
    ax.set_title("Detection Sensitivity across Attack Intensities", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_robustness_heatmap(
    noise_levels: List[str],
    missing_rates: List[str],
    f1_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot robustness performance under measurement noise and packet loss (Figure 8)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    cax = ax.matshow(f1_matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="F1-Score")

    ax.set_xticks(range(len(missing_rates)))
    ax.set_xticklabels(missing_rates)
    ax.set_yticks(range(len(noise_levels)))
    ax.set_yticklabels(noise_levels)

    # Annotate cells
    for i in range(len(noise_levels)):
        for j in range(len(missing_rates)):
            ax.text(j, i, f"{f1_matrix[i, j]:.3f}", ha="center", va="center", color="black" if f1_matrix[i, j] < 0.7 else "white", fontweight="bold")

    ax.set_xlabel("Missing Data Rate", fontsize=11)
    ax.set_ylabel("Measurement Noise Level", fontsize=11)
    ax.set_title("Detector Robustness to Telemetry Degradation", fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_localization_summary(
    model_loc_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot attack node localization performance (Top-k hits and Node F1) (Figure 9)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    models = list(model_loc_metrics.keys())
    top1 = [model_loc_metrics[m]["top1_hit"] for m in models]
    top3 = [model_loc_metrics[m]["top3_hit"] for m in models]
    node_f1 = [model_loc_metrics[m]["node_f1"] for m in models]

    x = np.arange(len(models))
    w = 0.25

    ax.bar(x - w, top1, width=w, label="Top-1 Hit Rate", color="#3182bd")
    ax.bar(x, top3, width=w, label="Top-3 Hit Rate", color="#6baed6")
    ax.bar(x + w, node_f1, width=w, label="Node F1 Score", color="#9ecae1")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Node-Level Attack Localization Accuracy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot performance across controlled ablation configurations (Figure 11)."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    variants = list(ablation_results.keys())
    auprc_vals = [ablation_results[v]["auprc"] for v in variants]
    f1_vals = [ablation_results[v]["f1"] for v in variants]

    x = np.arange(len(variants))
    w = 0.35

    ax.bar(x - w/2, auprc_vals, width=w, label="AUPRC", color="#2b8cbe")
    ax.bar(x + w/2, f1_vals, width=w, label="F1-Score", color="#feb24c")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Controlled Component Ablation Study (RQ5)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
