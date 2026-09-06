"""Smoke tests for all baseline and proposed anomaly detector forward passes."""

import numpy as np
import pytest
import torch

from gridguard.data.scaling import FeatureScaler
from gridguard.models.statistical import RobustStatisticalDetector
from gridguard.models.pca import PCADetector
from gridguard.models.isolation_forest import IsolationForestDetector
from gridguard.models.dense_ae import DenseAutoencoderDetector
from gridguard.models.lstm_ae import LSTMAutoencoderDetector
from gridguard.models.physics import PhysicsResidualDetector
from gridguard.models.graph_temporal import GraphTemporalDetector
from gridguard.models.proposed import ProposedPhysicsGraphTemporalDetector
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.ybus import YBusModel


@pytest.fixture
def dummy_data():
    N_samples = 20
    T_win = 4
    N_nodes = 4
    N_feat = 4

    X_win = np.random.normal(0, 1, size=(N_samples, T_win, N_nodes, N_feat)).astype(np.float32)
    Y_tar = np.random.normal(0, 1, size=(N_samples, N_nodes, N_feat)).astype(np.float32)
    adj = np.eye(N_nodes, dtype=np.float32)

    return X_win, Y_tar, adj


def test_models_smoke(dummy_data):
    X_win, Y_tar, adj = dummy_data
    N_samples, _, N_nodes, _ = X_win.shape

    # B0: Statistical MAD
    b0 = RobustStatisticalDetector().fit(Y_tar)
    g0, n0 = b0.score(Y_tar)
    assert g0.shape == (N_samples,)
    assert n0.shape == (N_samples, N_nodes)

    # B1: PCA SPE
    b1 = PCADetector(n_components=2).fit(Y_tar)
    g1, n1 = b1.score(Y_tar)
    assert g1.shape == (N_samples,)
    assert n1.shape == (N_samples, N_nodes)

    # B2: Isolation Forest
    b2 = IsolationForestDetector(n_estimators=10).fit(Y_tar)
    g2, n2 = b2.score(Y_tar)
    assert g2.shape == (N_samples,)
    assert n2.shape == (N_samples, N_nodes)

    # B3: Dense AE
    b3 = DenseAutoencoderDetector(hidden_dim=16, bottleneck_dim=8, epochs=1, batch_size=8).fit(Y_tar)
    g3, n3 = b3.score(Y_tar)
    assert g3.shape == (N_samples,)
    assert n3.shape == (N_samples, N_nodes)

    # B4: LSTM AE
    b4 = LSTMAutoencoderDetector(hidden_dim=16, epochs=1, batch_size=8).fit(X_win, Y_tar)
    g4, n4 = b4.score(X_win, Y_tar)
    assert g4.shape == (N_samples,)
    assert n4.shape == (N_samples, N_nodes)

    # B6: Graph-Temporal AE
    b6 = GraphTemporalDetector(adj_matrix=adj, graph_hidden_dim=16, graph_out_dim=16, gru_hidden_dim=16, epochs=1, batch_size=8).fit(X_win, Y_tar)
    g6, n6 = b6.score(X_win, Y_tar)
    assert g6.shape == (N_samples,)
    assert n6.shape == (N_samples, N_nodes)


def test_proposed_model_smoke():
    feeder_file = "tests/fixtures/tiny_feeder.dss"
    dss = OpenDSSInterface(feeder_file)
    ybus = YBusModel(dss, base_mva=1.0)
    N_nodes = ybus.num_nodes
    N_samples = 16
    T_win = 4
    N_feat = 4

    X_win = np.random.normal(0, 1, size=(N_samples, T_win, N_nodes, N_feat)).astype(np.float32)
    Y_tar = np.random.normal(0, 1, size=(N_samples, N_nodes, N_feat)).astype(np.float32)
    adj = np.eye(N_nodes, dtype=np.float32)

    scaler = FeatureScaler(["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"])
    scaler.means = {c: 0.0 for c in scaler.feature_cols}
    scaler.stds = {c: 1.0 for c in scaler.feature_cols}
    scaler.fitted = True

    m7 = ProposedPhysicsGraphTemporalDetector(
        adj_matrix=adj,
        ybus_model=ybus,
        scaler=scaler,
        graph_hidden_dim=16,
        graph_out_dim=16,
        gru_hidden_dim=16,
        epochs=1,
        batch_size=8,
    )
    m7.fit(X_win, Y_tar)

    v_mag = np.ones((len(X_win), N_nodes))
    v_ang = np.zeros((len(X_win), N_nodes))
    p_inj = np.zeros((len(X_win), N_nodes))
    q_inj = np.zeros((len(X_win), N_nodes))

    m7.fit_calibration_stats(X_win, Y_tar, v_mag, v_ang, p_inj, q_inj)
    g7, n7, rec, phys = m7.score(X_win, Y_tar, v_mag, v_ang, p_inj, q_inj)

    assert g7.shape == (len(X_win),)
    assert n7.shape == (len(X_win), ybus.num_nodes)
