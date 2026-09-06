"""Train proposed Physics-Guided Graph-Temporal Detector (B7) on clean training data."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch

from gridguard.data.splits import chronological_split
from gridguard.data.scaling import FeatureScaler
from gridguard.data.windows import create_sliding_windows
from gridguard.models.proposed import ProposedPhysicsGraphTemporalDetector
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.ybus import YBusModel
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed
from gridguard.utils.io import load_parquet, load_json, save_json

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Train proposed physics-guided graph-temporal detector")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts/models", help="Output directory")
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

    feat_cols = cfg.get("data", {}).get("features", ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"])
    scaler = FeatureScaler.from_dict(load_json(os.path.join(args.output_dir, "scaler.json")))

    df_train_scaled = scaler.transform_df(df_train)
    df_calib_scaled = scaler.transform_df(df_calib)

    win_len = cfg.get("data", {}).get("window_length", 8)
    X_tr_win, Y_tr_win, _, _, _ = create_sliding_windows(df_train_scaled, window_length=win_len, feature_cols=feat_cols)
    X_cal_win, Y_cal_win, _, _, _ = create_sliding_windows(df_calib_scaled, window_length=win_len, feature_cols=feat_cols)

    # Raw physical calibration tensors for physics consistency
    X_cal_raw, Y_cal_raw, _, _, _ = create_sliding_windows(df_calib, window_length=win_len, feature_cols=feat_cols)

    adj_path = os.path.join(args.data_dir, "adjacency_matrix.npy")
    adj_matrix = np.load(adj_path)

    feeder_file = sim_cfg.get("feeder_file", "tests/fixtures/tiny_feeder.dss")
    dss_iface = OpenDSSInterface(feeder_file)
    ybus_model = YBusModel(dss_iface, base_mva=cfg.get("data", {}).get("base_mva", 1.0))

    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    s_cfg = cfg.get("scoring", {})

    logger.info("Instantiating Proposed Model B7 (Physics-Guided GTAE)...")
    proposed_detector = ProposedPhysicsGraphTemporalDetector(
        adj_matrix=adj_matrix,
        ybus_model=ybus_model,
        scaler=scaler,
        graph_hidden_dim=m_cfg.get("graph_hidden_dim", 64),
        graph_out_dim=m_cfg.get("graph_out_dim", 64),
        gru_hidden_dim=m_cfg.get("gru_hidden_dim", 64),
        decoder_hidden_dim=m_cfg.get("decoder_hidden_dim", 64),
        dropout=m_cfg.get("dropout", 0.1),
        learning_rate=t_cfg.get("learning_rate", 0.001),
        weight_decay=t_cfg.get("weight_decay", 1e-5),
        batch_size=t_cfg.get("batch_size", 32),
        epochs=t_cfg.get("epochs", 5),
        lambda_physics=t_cfg.get("lambda_physics", 0.1),
        w_rec=s_cfg.get("w_rec", 0.5),
        w_phys=s_cfg.get("w_phys", 0.5),
        top_k=s_cfg.get("top_k", 3),
        device=cfg.get("experiment", {}).get("device", "cpu"),
    )

    logger.info("Training proposed model on clean training set...")
    proposed_detector.fit(X_tr_win, Y_tr_win)

    # Fit calibration statistics strictly on clean calibration set
    logger.info("Fitting calibration score normalizers on clean calibration set...")
    raw_V_mag = Y_cal_raw[:, :, 0]
    raw_V_ang = Y_cal_raw[:, :, 1]
    raw_P_inj = Y_cal_raw[:, :, 2]
    raw_Q_inj = Y_cal_raw[:, :, 3]

    proposed_detector.fit_calibration_stats(
        X_calib=X_cal_win,
        Y_calib=Y_cal_win,
        raw_V_mag=raw_V_mag,
        raw_V_ang=raw_V_ang,
        raw_P_inj=raw_P_inj,
        raw_Q_inj=raw_Q_inj,
    )

    save_path = os.path.join(args.output_dir, "proposed_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(proposed_detector, f)

    logger.info("Proposed model trained and saved successfully to: %s", save_path)


if __name__ == "__main__":
    main()
