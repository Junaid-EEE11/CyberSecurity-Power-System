"""Test node mapping and phase-aware graph construction."""

import pytest
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.feeder import FeederModel


def test_deterministic_node_mapping():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss1 = OpenDSSInterface(feeder_file)
    f1 = FeederModel(dss1)

    dss2 = OpenDSSInterface(feeder_file)
    f2 = FeederModel(dss2)

    assert f1.num_nodes == f2.num_nodes
    assert f1.node_names == f2.node_names
    assert len(f1.edge_index) == len(f2.edge_index)


def test_phase_aware_metadata():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss = OpenDSSInterface(feeder_file)
    feeder = FeederModel(dss)

    for meta in feeder.node_metadata:
        assert "node_id" in meta
        assert "bus_name" in meta
        assert "phase_num" in meta
        assert meta["phase_num"] in [1, 2, 3]

    adj = feeder.get_adjacency_matrix()
    assert adj.shape == (feeder.num_nodes, feeder.num_nodes)
    # Adjacency matrix should be symmetric for bidirectional branch connectivity
    assert (adj == adj.T).all()
