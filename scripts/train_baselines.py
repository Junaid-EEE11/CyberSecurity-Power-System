"""Train and fit baseline anomaly detectors (B0-B6) on clean training data."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch

from gridguard.data.splits import chronological_split
from gridguard.data.scaling import FeatureScaler
from gridguard.data.windows import create_sliding_windows
from gridguard.models.statistical import RobustStatisticalDetector
from gridguard.models.pca import PCADetector
from gridguard.models.isolation_forest import IsolationForestDetector
from gridguard.models.dense_ae import DenseAutoencoderDetector
from gridguard.models.lstm_ae import LSTMAutoencoderDetector
from gridguard.models.physics import PhysicsResidualDetector
from gridguard.models.graph_temporal import GraphTemporalDetector
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.ybus import YBusModel
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed
from gridguard.utils.io import load_parquet, save_json

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Train baseline anomaly detectors")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts/models", help="Model output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    dataset_path = os.path.join(args.data_dir, "telemetry_dataset.parquet")
    df = load_parquet(dataset_path)

    sim_cfg = cfg.get("simulation", {})
    split_cfg = sim_cfg.get("split_days", {"train": 3, "calibration": 1, "test": 1})
    res_min = sim_cfg.get("resolution_minutes", 15)
    steps_per_day = (24 * 60) // res_min

    df_train, df_calib, df_test = chronological_split(
        df,
        train_days=split_cfg["train"],
        calibration_days=split_cfg["calibration"],
        test_days=split_cfg["test"],
        steps_per_day=steps_per_day,
    )
    logger.info("Splits created: Train=%d, Calib=%d, Test=%d records", len(df_train), len(df_calib), len(df_test))

    # Fit Feature Scaler STRICTLY on Train split
    feat_cols = cfg.get("data", {}).get("features", ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"])
    scaler = FeatureScaler(feature_cols=feat_cols)
    scaler.fit(df_train)

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(scaler.to_dict(), os.path.join(args.output_dir, "scaler.json"))

    # Apply scaling
    df_train_scaled = scaler.transform_df(df_train)
    df_calib_scaled = scaler.transform_df(df_calib)

    win_len = cfg.get("data", {}).get("window_length", 8)
    X_tr_win, Y_tr_win, _, _, _ = create_sliding_windows(df_train_scaled, window_length=win_len, feature_cols=feat_cols)
    X_cal_win, Y_cal_win, _, _, _ = create_sliding_windows(df_calib_scaled, window_length=win_len, feature_cols=feat_cols)

    adj_path = os.path.join(args.data_dir, "adjacency_matrix.npy")
    adj_matrix = np.load(adj_path)

    epochs = cfg.get("training", {}).get("epochs", 5)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    device = cfg.get("experiment", {}).get("device", "cpu")

    models = {}

    # B0: Statistical MAD
    logger.info("Fitting B0 (Statistical MAD)...")
    b0 = RobustStatisticalDetector()
    b0.fit(Y_tr_win)
    models["B0_Statistical_MAD"] = b0

    # B1: PCA SPE
    logger.info("Fitting B1 (PCA SPE)...")
    b1 = PCADetector(seed=seed)
    b1.fit(Y_tr_win)
    models["B1_PCA_SPE"] = b1

    # B2: Isolation Forest
    logger.info("Fitting B2 (Isolation Forest)...")
    b2 = IsolationForestDetector(seed=seed)
    b2.fit(Y_tr_win)
    models["B2_IsolationForest"] = b2

    # B3: Dense AE
    logger.info("Fitting B3 (Dense AE)...")
    b3 = DenseAutoencoderDetector(epochs=epochs, batch_size=batch_size, device=device)
    b3.fit(Y_tr_win)
    models["B3_Dense_AE"] = b3

    # B4: LSTM AE
    logger.info("Fitting B4 (LSTM AE)...")
    b4 = LSTMAutoencoderDetector(epochs=epochs, batch_size=batch_size, device=device)
    b4.fit(X_tr_win, Y_tr_win)
    models["B4_LSTM_AE"] = b4

    # B5: Physics Residual Detector
    logger.info("Initializing B5 (Physics Residual Only)...")
    feeder_file = sim_cfg.get("feeder_file", "tests/fixtures/tiny_feeder.dss")
    dss_iface = OpenDSSInterface(feeder_file)
    ybus_model = YBusModel(dss_iface, base_mva=cfg.get("data", {}).get("base_mva", 1.0))
    b5 = PhysicsResidualDetector(ybus_model)
    models["B5_PhysicsResidualOnly"] = b5

    # B6: Graph-Temporal AE
    logger.info("Fitting B6 (Graph-Temporal AE)...")
    b6 = GraphTemporalDetector(adj_matrix=adj_matrix, epochs=epochs, batch_size=batch_size, device=device)
    b6.fit(X_tr_win, Y_tr_win)
    models["B6_GraphTemporal_AE"] = b6

    # Save baseline model instances
    save_path = os.path.join(args.output_dir, "baselines.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(models, f)

    logger.info("All baseline models fitted and saved to: %s", save_path)


if __name__ == "__main__":
    main()
