"""Attack A: Step False Data Injection Attack."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from gridguard.attacks.base import BaseAttack, AttackMetadata


class StepFDIA(BaseAttack):
    """Step FDIA: Injects a persistent constant relative bias into targeted bus-phase telemetry."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Attack_A_Step", seed=seed)

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

        end_step = start_step + duration - 1
        attack_id = f"step_{start_step}_{end_step}_str{int(strength*100)}"
        df = df_test.copy()

        # Identify rows to modify
        mask = (
            (df["timestep"] >= start_step)
            & (df["timestep"] <= end_step)
            & (df["node_id"].isin(target_nodes))
        )

        # Random sign (+ or -) per node/feature
        for node in target_nodes:
            node_mask = mask & (df["node_id"] == node)
            sign = self.rng.choice([-1.0, 1.0])
            for feat in target_features:
                clean_col = f"clean_{feat}"
                if clean_col in df.columns:
                    base_vals = df.loc[node_mask, clean_col]
                else:
                    base_vals = df.loc[node_mask, feat]

                # Step perturbation: V_att = V_clean * (1 + sign * strength)
                # or for small magnitude values, add offset
                if "v_mag" in feat:
                    df.loc[node_mask, feat] = base_vals * (1.0 + sign * strength)
                else:
                    df.loc[node_mask, feat] = base_vals + sign * strength

        # Set attack labels and ground truth
        df.loc[mask, "is_attack"] = 1
        df.loc[mask, "attack_id"] = attack_id
        df.loc[mask, "attack_type"] = self.name
        df.loc[mask, "attack_strength"] = float(strength)
        df.loc[mask, "is_compromised"] = 1

        # Also mark global attack indicator for all nodes at that timestep
        timestep_mask = (df["timestep"] >= start_step) & (df["timestep"] <= end_step)
        df.loc[timestep_mask, "is_attack"] = 1

        target_node_names = []
        if extra_context and "node_names" in extra_context:
            target_node_names = [extra_context["node_names"][i] for i in target_nodes if i < len(extra_context["node_names"])]

        metadata = AttackMetadata(
            attack_id=attack_id,
            attack_type=self.name,
            start_timestep=start_step,
            end_timestep=end_step,
            target_nodes=target_nodes,
            target_node_names=target_node_names,
            target_features=target_features,
            attack_strength=float(strength),
            seed=self.seed,
        )

        return df, metadata
