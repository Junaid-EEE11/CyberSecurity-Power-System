"""Test OpenDSS interface wrapper and feeder solver."""

import pytest
from gridguard.simulation.opendss_interface import OpenDSSInterface


def test_opendss_interface_solves():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss_iface = OpenDSSInterface(feeder_file)
    converged = dss_iface.solve()
    assert converged, "OpenDSS failed to converge on tiny feeder fixture"

    node_names = dss_iface.get_all_node_names()
    assert len(node_names) > 0, "No nodes returned from OpenDSS"
    assert "sourcebus.1" in node_names
    assert "bus2.1" in node_names


def test_opendss_loads_and_losses():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss_iface = OpenDSSInterface(feeder_file)
    dss_iface.solve()

    loads = dss_iface.get_load_names()
    assert len(loads) >= 2
    assert any("load2" in l for l in loads)
    assert any("load3" in l for l in loads)

    losses = dss_iface.get_total_losses()
    assert losses[0] > 0.0, "Active losses must be strictly positive"
