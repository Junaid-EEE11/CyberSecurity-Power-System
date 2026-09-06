"""Feeder topology and graph network visualization."""

import os
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx

from gridguard.simulation.feeder import FeederModel


def plot_feeder_topology(feeder_model: FeederModel, save_path: Optional[str] = None) -> plt.Figure:
    """Generate publication-ready plot of unbalanced distribution feeder graph.

    Args:
        feeder_model: FeederModel containing network graph and node metadata.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    G = feeder_model.graph

    # Node coloring by phase
    phase_colors = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", "A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a"}
    node_colors = []
    for n in G.nodes():
        p = G.nodes[n].get("phase_num", 1)
        node_colors.append(phase_colors.get(p, "#999999"))

    # Layout using spring layout with deterministic seed
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=35, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#888888", width=0.75, alpha=0.6, ax=ax)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="Phase A", markerfacecolor="#e41a1c", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Phase B", markerfacecolor="#377eb8", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Phase C", markerfacecolor="#4daf4a", markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)
    ax.set_title(f"Unbalanced Feeder Graph Topology ({feeder_model.num_nodes} Bus-Phase Nodes)", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
