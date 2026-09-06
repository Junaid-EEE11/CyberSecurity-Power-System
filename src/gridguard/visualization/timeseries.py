"""Time-series telemetry and attack visualization."""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_clean_timeseries(
    df: pd.DataFrame,
    node_ids: List[int],
    start_step: int = 0,
    num_steps: int = 192,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multi-day multivariate clean telemetry curves (Figure 3)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, dpi=300)

    subset = df[(df["timestep"] >= start_step) & (df["timestep"] < start_step + num_steps)]

    for node in node_ids:
        node_df = subset[subset["node_id"] == node]
        t = node_df["timestep"] - start_step
        axes[0].plot(t, node_df["v_mag_pu"], label=f"Node {node}", alpha=0.85, linewidth=1.2)
        axes[1].plot(t, node_df["p_inj_pu"], label=f"Node {node}", alpha=0.85, linewidth=1.2)
        axes[2].plot(t, node_df["q_inj_pu"], label=f"Node {node}", alpha=0.85, linewidth=1.2)

    axes[0].set_ylabel("Voltage (p.u.)", fontsize=10)
    axes[1].set_ylabel("Active Power P (p.u.)", fontsize=10)
    axes[2].set_ylabel("Reactive Power Q (p.u.)", fontsize=10)
    axes[2].set_xlabel("Time Index (15-min intervals)", fontsize=10)

    axes[0].set_title("Multivariate Operational Telemetry over 48 Hours", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", ncol=len(node_ids), fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_attack_comparison(
    df: pd.DataFrame,
    attack_id: str,
    target_node: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot clean vs attacked measurement trajectories across attack event (Figure 4)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)

    att_rows = df[df["attack_id"] == attack_id]
    if att_rows.empty:
        start_t = 50
        end_t = 80
    else:
        start_t = att_rows["timestep"].min() - 10
        end_t = att_rows["timestep"].max() + 10

    sub = df[(df["timestep"] >= start_t) & (df["timestep"] <= end_t) & (df["node_id"] == target_node)]

    t = sub["timestep"]
    clean_v = sub["clean_v_mag_pu"] if "clean_v_mag_pu" in sub.columns else sub["v_mag_pu"]
    att_v = sub["v_mag_pu"]

    clean_p = sub["clean_p_inj_pu"] if "clean_p_inj_pu" in sub.columns else sub["p_inj_pu"]
    att_p = sub["p_inj_pu"]

    axes[0].plot(t, clean_v, "k--", label="Ground Truth Clean", linewidth=1.5)
    axes[0].plot(t, att_v, "r-", label="Attacked Telemetry", linewidth=1.5)
    axes[0].set_ylabel("Voltage (p.u.)", fontsize=10)
    axes[0].set_title(f"Telemetry Manipulation Profile: Node {target_node}", fontsize=12, fontweight="bold")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t, clean_p, "k--", label="Ground Truth Clean", linewidth=1.5)
    axes[1].plot(t, att_p, "r-", label="Attacked Telemetry", linewidth=1.5)
    axes[1].set_ylabel("Active Power (p.u.)", fontsize=10)
    axes[1].set_xlabel("Timestep", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
