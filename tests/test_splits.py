"""Test chronological dataset splitting and temporal leakage prevention."""

import pandas as pd
import pytest
from gridguard.data.splits import chronological_split


def test_chronological_splitting_no_leakage():
    # Synthetic dataframe with 10 days, 96 steps/day = 960 steps
    steps = 960
    data = []
    for s in range(steps):
        for n in range(4):
            data.append({"timestep": s, "node_id": n, "v_mag_pu": 1.0 + 0.01 * s})
    df = pd.DataFrame(data)

    df_train, df_calib, df_test = chronological_split(
        df,
        train_days=6,
        calibration_days=2,
        test_days=2,
        steps_per_day=96,
    )

    # Train: steps [0, 575]
    assert df_train["timestep"].min() == 0
    assert df_train["timestep"].max() == 575

    # Calib: steps [576, 767]
    assert df_calib["timestep"].min() == 576
    assert df_calib["timestep"].max() == 767

    # Test: steps [768, 959]
    assert df_test["timestep"].min() == 768
    assert df_test["timestep"].max() == 959

    # Zero overlap
    train_steps = set(df_train["timestep"])
    calib_steps = set(df_calib["timestep"])
    test_steps = set(df_test["timestep"])

    assert len(train_steps.intersection(calib_steps)) == 0
    assert len(train_steps.intersection(test_steps)) == 0
    assert len(calib_steps.intersection(test_steps)) == 0
