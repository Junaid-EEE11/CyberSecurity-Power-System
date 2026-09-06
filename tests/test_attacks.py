"""Test cyberattack injection generators and metadata tracking."""

import pandas as pd
import numpy as np
import pytest
from gridguard.attacks.step import StepFDIA
from gridguard.attacks.ramp import RampFDIA
from gridguard.attacks.replay import ReplayFDIA


def test_step_attack_modifies_only_targets_and_window():
    records = []
    for t in range(30):
        for n in range(5):
            records.append({
                "timestep": t,
                "node_id": n,
                "v_mag_pu": 1.0,
                "p_inj_pu": 0.5,
                "q_inj_pu": 0.1,
                "clean_v_mag_pu": 1.0,
                "clean_p_inj_pu": 0.5,
                "clean_q_inj_pu": 0.1,
                "is_attack": 0,
                "is_compromised": 0,
                "attack_id": "none",
                "attack_type": "none",
                "attack_strength": 0.0,
            })
    df = pd.DataFrame(records)

    attacker = StepFDIA()
    df_att, meta = attacker.apply(
        df_test=df,
        start_step=10,
        duration=11,  # steps 10 to 20
        target_nodes=[1, 3],
        strength=0.10,
        target_features=["v_mag_pu"],
    )

    # Check non-attacked times unchanged
    assert (df_att[df_att["timestep"] < 10]["v_mag_pu"] == 1.0).all()
    assert (df_att[df_att["timestep"] > 20]["v_mag_pu"] == 1.0).all()

    # Check non-target nodes unchanged
    assert (df_att[(df_att["timestep"] >= 10) & (df_att["timestep"] <= 20) & (df_att["node_id"] == 0)]["v_mag_pu"] == 1.0).all()

    # Check attacked targets modified
    att_rows = df_att[(df_att["timestep"] >= 10) & (df_att["timestep"] <= 20) & (df_att["node_id"].isin([1, 3]))]
    assert np.all(np.isin(np.round(att_rows["v_mag_pu"].values, 2), [0.90, 1.10]))
    assert (att_rows["is_compromised"] == 1).all()
    assert (att_rows["is_attack"] == 1).all()


def test_replay_attack_uses_only_past_data():
    records = []
    for t in range(50):
        for n in range(2):
            records.append({
                "timestep": t,
                "node_id": n,
                "v_mag_pu": float(t),
                "clean_v_mag_pu": float(t),
                "p_inj_pu": 0.1,
                "clean_p_inj_pu": 0.1,
                "q_inj_pu": 0.1,
                "clean_q_inj_pu": 0.1,
                "is_attack": 0,
                "is_compromised": 0,
                "attack_id": "none",
                "attack_type": "none",
                "attack_strength": 0.0,
            })
    df = pd.DataFrame(records)

    attacker = ReplayFDIA()
    df_att, meta = attacker.apply(
        df_test=df,
        start_step=30,
        duration=10,
        target_nodes=[0],
        strength=20.0,  # 20 steps lag
        target_features=["v_mag_pu"],
    )

    sub = df_att[(df_att["timestep"] == 30) & (df_att["node_id"] == 0)]
    assert np.isclose(sub["v_mag_pu"].iloc[0], 10.0), "Replayed value must come from past lag step"
