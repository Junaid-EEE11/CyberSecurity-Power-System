"""Test Kirchhoff power flow equation consistency: S = V * conj(Y * V)."""

import numpy as np
import pytest
import torch

from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.feeder import FeederModel
from gridguard.simulation.measurements import extract_measurements
from gridguard.simulation.ybus import YBusModel


def test_power_consistency_calculation():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss = OpenDSSInterface(feeder_file)
    feeder = FeederModel(dss)
    ybus = YBusModel(dss, base_mva=1.0)

    dss.solve()
    meas = extract_measurements(dss, feeder, base_mva=1.0)

    # Calculate residual on clean converged state
    res = ybus.calculate_physics_residual_numpy(
        V_mag=meas["v_mag_pu"],
        V_ang_rad=meas["v_ang_rad"],
        P_inj_pu=meas["p_inj_pu"],
        Q_inj_pu=meas["q_inj_pu"],
    )

    # For clean state, power balance error should be small
    assert np.mean(res) < 0.25, f"Clean power flow consistency error too high: {np.mean(res)}"


def test_torch_physics_layer_gradient():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss = OpenDSSInterface(feeder_file)
    ybus = YBusModel(dss, base_mva=1.0)
    layer = ybus.get_torch_physics_layer("cpu")

    N = ybus.num_nodes
    v_real = (torch.ones(2, N) * 2400.0).requires_grad_()
    v_imag = torch.zeros(2, N, requires_grad=True)
    p_inj = torch.zeros(2, N, requires_grad=True)
    q_inj = torch.zeros(2, N, requires_grad=True)

    loss = layer(v_real, v_imag, p_inj, q_inj)
    assert loss.item() >= 0.0
    loss.backward()

    assert v_real.grad is not None
    assert not torch.isnan(v_real.grad).any()
