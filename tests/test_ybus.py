"""Test System Admittance Y-matrix extraction and properties."""

import numpy as np
import pytest
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.ybus import YBusModel


def test_ybus_matrix_properties():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss = OpenDSSInterface(feeder_file)
    ybus_model = YBusModel(dss, base_mva=1.0)

    assert ybus_model.num_nodes > 0
    Y_dense = ybus_model.get_dense_ybus()
    assert Y_dense.shape == (ybus_model.num_nodes, ybus_model.num_nodes)

    # Complex admittance matrix of passive reciprocal grid is symmetric: Y = Y^T
    np.testing.assert_allclose(Y_dense, Y_dense.T, atol=1e-6)
    # Diagonal elements typically have positive conductance (real part > 0) or small shunt
    assert np.all(np.real(np.diag(Y_dense)) >= -1e-6)
