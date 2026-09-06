"""Attack E: Physics-Plausible Operating-Point Substitution Attack (Unseen Attack)."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from gridguard.attacks.base import BaseAttack, AttackMetadata


class OperatingPointSubstitutionFDIA(BaseAttack):
    """Attack E: Substitutes measurements from an alternative converged power flow solution."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Attack_E_OperatingPoint", seed=seed)

    def apply(
        self,
        df_test: pd.DataFrame,
        start_step: int,
        duration: int,
        target_nodes: List[int],
        strength: float = 0.20,  # strength indicates load perturbation scale (e.g. 20%)
        target_features: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, AttackMetadata]:
        """Apply operating point substitution attack.

        If an OpenDSS simulator instance is provided in extra_context, re-solve power flow
        with perturbed loads and inject the physical converged state.
        Otherwise, compute a physically consistent linearized power flow substitution.
        """
        if target_features is None:
            target_features = ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu", "sin_theta", "cos_theta"]

        end_step = start_step + duration - 1
        attack_id = f"op_sub_{start_step}_{end_step}_str{int(strength*100)}"
        df = df_test.copy()

        # If OpenDSS simulator is provided
        dss_interface = extra_context.get("dss_interface") if extra_context else None
        feeder_model = extra_context.get("feeder_model") if extra_context else None

        for step in range(start_step, end_step + 1):
            # Check if we can do full OpenDSS re-solve
            if dss_interface is not None and feeder_model is not None and "load_profiles" in extra_context:
                load_profiles = extra_context["load_profiles"]
                load_names = dss_interface.get_load_names()
                nominal_loads = extra_context.get("nominal_loads", {})

                # Perturb underlying loads
                perturb_factor = 1.0 + self.rng.uniform(-strength, strength)
                for l_idx, l_name in enumerate(load_names):
                    if l_name in nominal_loads:
                        base_kw = nominal_loads[l_name]["kw"]
                        base_kvar = nominal_loads[l_name]["kvar"]
                        mult = load_profiles[step, l_idx] * perturb_factor
                        dss_interface.set_individual_load(l_name, kw=base_kw * mult, kvar=base_kvar * mult)

                dss_interface.solve()
                # Extract new converged physical state
                from gridguard.simulation.measurements import extract_measurements
                alt_meas = extract_measurements(dss_interface, feeder_model, base_mva=extra_context.get("base_mva", 1.0))

                for node in target_nodes:
                    row_idx = df[(df["timestep"] == step) & (df["node_id"] == node)].index
                    for feat in target_features:
                        if feat in alt_meas:
                            df.loc[row_idx, feat] = alt_meas[feat][node]

                    df.loc[row_idx, "is_compromised"] = 1
                    df.loc[row_idx, "attack_id"] = attack_id
                    df.loc[row_idx, "attack_type"] = self.name
                    df.loc[row_idx, "attack_strength"] = float(strength)
            else:
                # Physically plausible voltage/power coupled substitution
                dv_mult = 1.0 + self.rng.choice([-1.0, 1.0]) * strength * 0.5
                dp_mult = 1.0 + self.rng.choice([-1.0, 1.0]) * strength

                for node in target_nodes:
                    row_idx = df[(df["timestep"] == step) & (df["node_id"] == node)].index
                    clean_v = df.loc[row_idx, "clean_v_mag_pu"].values[0] if "clean_v_mag_pu" in df.columns else df.loc[row_idx, "v_mag_pu"].values[0]
                    clean_p = df.loc[row_idx, "clean_p_inj_pu"].values[0] if "clean_p_inj_pu" in df.columns else df.loc[row_idx, "p_inj_pu"].values[0]
                    clean_q = df.loc[row_idx, "clean_q_inj_pu"].values[0] if "clean_q_inj_pu" in df.columns else df.loc[row_idx, "q_inj_pu"].values[0]

                    df.loc[row_idx, "v_mag_pu"] = clean_v * dv_mult
                    df.loc[row_idx, "p_inj_pu"] = clean_p * dp_mult
                    df.loc[row_idx, "q_inj_pu"] = clean_q * dp_mult
                    df.loc[row_idx, "is_compromised"] = 1
                    df.loc[row_idx, "attack_id"] = attack_id
                    df.loc[row_idx, "attack_type"] = self.name
                    df.loc[row_idx, "attack_strength"] = float(strength)

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
            extra_info={"is_unseen": True},
        )

        return df, metadata
