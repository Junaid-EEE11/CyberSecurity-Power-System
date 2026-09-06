"""Attack D: Coordinated Spatial False Data Injection Attack on connected subgraphs."""

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from gridguard.attacks.base import BaseAttack, AttackMetadata


class CoordinatedSpatialFDIA(BaseAttack):
    """Coordinated Spatial FDIA: Manipulates a connected topological subgraph/neighborhood simultaneously."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Attack_D_Coordinated", seed=seed)

    def select_subgraph_nodes(
        self, graph: nx.Graph, center_node: int, max_nodes: int = 6, radius: int = 2
    ) -> List[int]:
        """Select a connected local neighborhood around a center node."""
        subgraph_nodes = set([center_node])
        frontier = [center_node]

        for _ in range(radius):
            next_frontier = []
            for u in frontier:
                for v in graph.neighbors(u):
                    if v not in subgraph_nodes:
                        subgraph_nodes.add(v)
                        next_frontier.append(v)
                        if len(subgraph_nodes) >= max_nodes:
                            return list(subgraph_nodes)
            frontier = next_frontier

        return list(subgraph_nodes)

    def apply(
        self,
        df_test: pd.DataFrame,
        start_step: int,
        duration: int,
        target_nodes: List[int],
        strength: float,
        target_features: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, AttackMetadata]:
        if target_features is None:
            target_features = ["v_mag_pu", "p_inj_pu", "q_inj_pu"]

        # If graph is provided in context, select a coherent spatial cluster
        if extra_context and "graph" in extra_context:
            G = extra_context["graph"]
            center = target_nodes[0] if len(target_nodes) > 0 else 0
            connected_targets = self.select_subgraph_nodes(
                G, center_node=center, max_nodes=max(3, len(target_nodes))
            )
        else:
            connected_targets = target_nodes

        end_step = start_step + duration - 1
        attack_id = f"coord_{start_step}_{end_step}_str{int(strength*100)}"
        df = df_test.copy()

        # In coordinated attack, manipulate all nodes in cluster with consistent spatial direction
        cluster_sign = self.rng.choice([-1.0, 1.0])

        mask = (
            (df["timestep"] >= start_step)
            & (df["timestep"] <= end_step)
            & (df["node_id"].isin(connected_targets))
        )

        for feat in target_features:
            clean_col = f"clean_{feat}"
            base_vals = df.loc[mask, clean_col] if clean_col in df.columns else df.loc[mask, feat]

            if "v_mag" in feat:
                df.loc[mask, feat] = base_vals * (1.0 + cluster_sign * strength)
            else:
                df.loc[mask, feat] = base_vals + cluster_sign * strength

        df.loc[mask, "is_attack"] = 1
        df.loc[mask, "attack_id"] = attack_id
        df.loc[mask, "attack_type"] = self.name
        df.loc[mask, "attack_strength"] = float(strength)
        df.loc[mask, "is_compromised"] = 1

        # Global flag
        timestep_mask = (df["timestep"] >= start_step) & (df["timestep"] <= end_step)
        df.loc[timestep_mask, "is_attack"] = 1

        target_node_names = []
        if extra_context and "node_names" in extra_context:
            target_node_names = [extra_context["node_names"][i] for i in connected_targets if i < len(extra_context["node_names"])]

        metadata = AttackMetadata(
            attack_id=attack_id,
            attack_type=self.name,
            start_timestep=start_step,
            end_timestep=end_step,
            target_nodes=connected_targets,
            target_node_names=target_node_names,
            target_features=target_features,
            attack_strength=float(strength),
            seed=self.seed,
            extra_info={"subgraph_size": len(connected_targets)},
        )

        return df, metadata
