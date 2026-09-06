"""Test sliding temporal window extraction and boundary preservation."""

import numpy as np
import pandas as pd
import pytest
from gridguard.data.windows import create_sliding_windows


def test_sliding_window_dimensions():
    # 50 timesteps, 4 nodes, 3 features
    records = []
    for t in range(50):
        for n in range(4):
            records.append({
                "timestep": t,
                "node_id": n,
                "f1": float(t),
                "f2": float(n),
                "f3": 1.0,
                "is_attack": int(t >= 40),
                "is_compromised": int(t >= 40 and n == 1),
            })
    df = pd.DataFrame(records)

    win_len = 8
    X, Y, y_global, y_nodes, steps = create_sliding_windows(
        df,
        window_length=win_len,
        feature_cols=["f1", "f2", "f3"],
    )

    expected_windows = 50 - win_len + 1  # 43 windows
    assert X.shape == (expected_windows, win_len, 4, 3)
    assert Y.shape == (expected_windows, 4, 3)
    assert y_global.shape == (expected_windows,)
    assert y_nodes.shape == (expected_windows, 4)
    assert len(steps) == expected_windows

    # Verify target matches last element of window
    np.testing.assert_allclose(X[:, -1, :, :], Y)
