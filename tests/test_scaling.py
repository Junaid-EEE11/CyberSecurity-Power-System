"""Test feature standardization fitting strictly on training split."""

import numpy as np
import pandas as pd
import pytest
from gridguard.data.scaling import FeatureScaler


def test_scaler_fit_strictly_on_train():
    df_train = pd.DataFrame({
        "v_mag_pu": [0.95, 1.00, 1.05],
        "p_inj_pu": [-0.10, -0.20, -0.30],
    })
    df_test = pd.DataFrame({
        "v_mag_pu": [1.50, 2.00],  # Out of distribution
        "p_inj_pu": [0.50, 1.00],
    })

    scaler = FeatureScaler(["v_mag_pu", "p_inj_pu"])
    scaler.fit(df_train)

    assert np.isclose(scaler.means["v_mag_pu"], 1.00)
    assert np.isclose(scaler.means["p_inj_pu"], -0.20)

    # Transform test set using train params
    df_test_scaled = scaler.transform_df(df_test)
    assert not np.isnan(df_test_scaled["v_mag_pu"]).any()

    # Inverse transform
    arr_inv = scaler.inverse_transform(df_test_scaled[["v_mag_pu", "p_inj_pu"]].values)
    np.testing.assert_allclose(arr_inv, df_test[["v_mag_pu", "p_inj_pu"]].values, atol=1e-5)
