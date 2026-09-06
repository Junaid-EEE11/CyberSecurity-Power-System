"""Attack B: Ramp (Low-and-Slow) False Data Injection Attack."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from gridguard.attacks.base import BaseAttack, AttackMetadata


class RampFDIA(BaseAttack):
    """Ramp FDIA: Gradually ramps up bias over time to evade sudden-jump residual detectors."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Attack_B_Ramp", seed=seed)

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
        attack_id = f"ramp_{start_step}_{end_step}_str{int(strength*100)}"
        df = df_test.copy()

        # Generate ramp weights from 0.0 to 1.0 across the duration
        ramp_factors = np.linspace(0.1, 1.0, duration)

        for step_idx, step in enumerate(range(start_step, end_step + 1)):
            factor = ramp_factors[step_idx]
            step_strength = strength * factor

            for node in target_nodes:
                row_mask = (df["timestep"] == step) & (df["node_id"] == node)
                sign = self.rng.choice([-1.0, 1.0])

                for feat in target_features:
                    clean_col = f"clean_{feat}"
                    base_val = df.loc[row_mask, clean_col].values[0] if clean_col in df.columns else df.loc[row_mask, feat].values[0]

                    if "v_mag" in feat:
                        df.loc[row_mask, feat] = base_val * (1.0 + sign * step_strength)
                    else:
                        df.loc[row_mask, feat] = base_val + sign * step_strength

                df.loc[row_mask, "is_compromised"] = 1
                df.loc[row_mask, "attack_id"] = attack_id
                df.loc[row_mask, "attack_type"] = self.name
                df.loc[row_mask, "attack_strength"] = float(step_strength)

            # Global timestep attack indicator
            df.loc[df["timestep"] == step, "is_attack"] = 1

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
            extra_info={"ramp_duration": duration},
        )

        return df, metadata
