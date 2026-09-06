"""Attack C: Replay / Stale-Data Cyberattack."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from gridguard.attacks.base import BaseAttack, AttackMetadata


class ReplayFDIA(BaseAttack):
    """Replay Attack: Substitutes real-time measurements with past legitimate telemetry from lag window."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Attack_C_Replay", seed=seed)

    def apply(
        self,
        df_test: pd.DataFrame,
        start_step: int,
        duration: int,
        target_nodes: List[int],
        strength: float = 0.0,  # strength indicates lag days or steps
        target_features: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, AttackMetadata]:
        if target_features is None:
            target_features = ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu", "sin_theta", "cos_theta"]

        end_step = start_step + duration - 1
        lag_steps = int(strength) if strength >= 12 else 48  # Default lag: 48 steps (12 hours)

        historical_start = start_step - lag_steps
        historical_end = end_step - lag_steps

        # Ensure historical window exists in df
        min_step = df_test["timestep"].min()
        if historical_start < min_step:
            lag_steps = max(12, start_step - min_step)
            historical_start = start_step - lag_steps
            historical_end = end_step - lag_steps

        attack_id = f"replay_{start_step}_{end_step}_lag{lag_steps}"
        df = df_test.copy()

        for step_offset in range(duration):
            curr_step = start_step + step_offset
            past_step = historical_start + step_offset

            for node in target_nodes:
                past_row = df[(df["timestep"] == past_step) & (df["node_id"] == node)]
                curr_row_idx = df[(df["timestep"] == curr_step) & (df["node_id"] == node)].index

                if not past_row.empty and len(curr_row_idx) > 0:
                    for feat in target_features:
                        if feat in past_row.columns:
                            val = past_row[feat].values[0]
                            df.loc[curr_row_idx, feat] = val

                    df.loc[curr_row_idx, "is_compromised"] = 1
                    df.loc[curr_row_idx, "attack_id"] = attack_id
                    df.loc[curr_row_idx, "attack_type"] = self.name
                    df.loc[curr_row_idx, "attack_strength"] = float(lag_steps)

            # Global attack flag
            df.loc[df["timestep"] == curr_step, "is_attack"] = 1

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
            attack_strength=float(lag_steps),
            seed=self.seed,
            extra_info={"lag_steps": lag_steps, "historical_start": historical_start},
        )

        return df, metadata
