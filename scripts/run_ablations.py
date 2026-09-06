"""Run controlled component ablation studies (A1 - A6)."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from gridguard.calibration.conformal import ConformalCalibrator
from gridguard.data.splits import chronological_split
from gridguard.data.scaling import FeatureScaler
from gridguard.data.windows import create_sliding_windows
from gridguard.evaluation.detection import compute_detection_metrics
from gridguard.models.proposed import ProposedPhysicsGraphTemporalDetector
from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.ybus import YBusModel
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed
from gridguard.utils.io import load_parquet, load_json, save_json

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing dataset")
    parser.add_argument("--models_dir", type=str, default="artifacts/models", help="Directory containing models")
    parser.add_argument("--output_dir", type=str, default="artifacts/ablations", help="Ablation output directory")
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
    scaler = FeatureScaler.from_dict(load_json(os.path.join(args.models_dir, "scaler.json")))

    df_train_scaled = scaler.transform_df(df_train)
    df_calib_scaled = scaler.transform_df(df_calib)
    df_test_scaled = scaler.transform_df(df_test)

    win_len = cfg.get("data", {}).get("window_length", 8)
    X_tr_win, Y_tr_win, _, _, _ = create_sliding_windows(df_train_scaled, window_length=win_len, feature_cols=feat_cols)
    X_cal_win, Y_cal_win, _, _, _ = create_sliding_windows(df_calib_scaled, window_length=win_len, feature_cols=feat_cols)
    X_test_win, Y_test_win, y_test_global, _, _ = create_sliding_windows(df_test_scaled, window_length=win_len, feature_cols=feat_cols)

    X_cal_raw, Y_cal_raw, _, _, _ = create_sliding_windows(df_calib, window_length=win_len, feature_cols=feat_cols)
    X_test_raw, Y_test_raw, _, _, _ = create_sliding_windows(df_test, window_length=win_len, feature_cols=feat_cols)

    adj_path = os.path.join(args.data_dir, "adjacency_matrix.npy")
    adj_matrix = np.load(adj_path)

    feeder_file = sim_cfg.get("feeder_file", "tests/fixtures/tiny_feeder.dss")
    dss_iface = OpenDSSInterface(feeder_file)
    ybus_model = YBusModel(dss_iface, base_mva=cfg.get("data", {}).get("base_mva", 1.0))

    # Load baseline proposed model
    with open(os.path.join(args.models_dir, "proposed_model.pkl"), "rb") as f:
        proposed_base = pickle.load(f)

    # Base full proposed evaluation
    cal_scores, _, _, _ = proposed_base.score(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    test_scores, _, _, _ = proposed_base.score(X_test_win, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])

    calibrator = ConformalCalibrator(alpha_values=[0.05])
    calibrator.fit(cal_scores)
    thresh = calibrator.thresholds[0.05]
    base_metrics = compute_detection_metrics(y_test_global, test_scores, threshold=thresh)

    ablation_results = {
        "Full_Proposed_B7": {
            "description": "Full proposed model (GraphSAGE + GRU + Physics Loss + Fused Score)",
            "auroc": base_metrics["auroc"],
            "auprc": base_metrics["auprc"],
            "f1": base_metrics["f1"],
            "far": base_metrics["far"],
        }
    }

    # A1: Remove Graph Structure (identity adjacency)
    logger.info("Running Ablation A1: Remove Graph Structure...")
    eye_adj = np.eye(adj_matrix.shape[0])
    m_a1 = ProposedPhysicsGraphTemporalDetector(
        adj_matrix=eye_adj,
        ybus_model=ybus_model,
        scaler=scaler,
        epochs=3,
        batch_size=32,
    )
    m_a1.fit(X_tr_win, Y_tr_win)
    m_a1.fit_calibration_stats(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    c_s, _, _, _ = m_a1.score(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    t_s, _, _, _ = m_a1.score(X_test_win, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])
    cal_a1 = ConformalCalibrator(alpha_values=[0.05]).fit(c_s)
    res_a1 = compute_detection_metrics(y_test_global, t_s, threshold=cal_a1.thresholds[0.05])
    ablation_results["A1_No_Graph"] = {
        "description": "No graph message passing (identity topology)",
        "auroc": res_a1["auroc"],
        "auprc": res_a1["auprc"],
        "f1": res_a1["f1"],
        "far": res_a1["far"],
    }

    # A2: Remove Temporal Modeling (window length = 1)
    logger.info("Running Ablation A2: Remove Temporal Modeling...")
    X_tr_win1 = Y_tr_win[:, np.newaxis, :, :]
    X_cal_win1 = Y_cal_win[:, np.newaxis, :, :]
    X_test_win1 = Y_test_win[:, np.newaxis, :, :]
    m_a2 = ProposedPhysicsGraphTemporalDetector(
        adj_matrix=adj_matrix,
        ybus_model=ybus_model,
        scaler=scaler,
        epochs=3,
        batch_size=32,
    )
    m_a2.fit(X_tr_win1, Y_tr_win)
    m_a2.fit_calibration_stats(X_cal_win1, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    c_s, _, _, _ = m_a2.score(X_cal_win1, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    t_s, _, _, _ = m_a2.score(X_test_win1, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])
    cal_a2 = ConformalCalibrator(alpha_values=[0.05]).fit(c_s)
    res_a2 = compute_detection_metrics(y_test_global, t_s, threshold=cal_a2.thresholds[0.05])
    ablation_results["A2_No_Temporal"] = {
        "description": "Static snapshot only (T=1, no temporal history)",
        "auroc": res_a2["auroc"],
        "auprc": res_a2["auprc"],
        "f1": res_a2["f1"],
        "far": res_a2["far"],
    }

    # A3: Remove Physics Loss (lambda_physics = 0)
    logger.info("Running Ablation A3: Remove Physics Loss...")
    m_a3 = ProposedPhysicsGraphTemporalDetector(
        adj_matrix=adj_matrix,
        ybus_model=ybus_model,
        scaler=scaler,
        lambda_physics=0.0,
        epochs=3,
        batch_size=32,
    )
    m_a3.fit(X_tr_win, Y_tr_win)
    m_a3.fit_calibration_stats(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    c_s, _, _, _ = m_a3.score(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    t_s, _, _, _ = m_a3.score(X_test_win, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])
    cal_a3 = ConformalCalibrator(alpha_values=[0.05]).fit(c_s)
    res_a3 = compute_detection_metrics(y_test_global, t_s, threshold=cal_a3.thresholds[0.05])
    ablation_results["A3_No_Physics_Loss"] = {
        "description": "Trained without electrical physics loss (lambda_phys=0)",
        "auroc": res_a3["auroc"],
        "auprc": res_a3["auprc"],
        "f1": res_a3["f1"],
        "far": res_a3["far"],
    }

    # A4: Conventional Percentile Threshold (instead of Conformal)
    logger.info("Running Ablation A4: Percentile Threshold...")
    raw_perc_thresh = float(np.percentile(cal_scores, 95.0))
    res_a4 = compute_detection_metrics(y_test_global, test_scores, threshold=raw_perc_thresh)
    ablation_results["A4_Conventional_Percentile"] = {
        "description": "Conventional 95th percentile threshold without conformal correction",
        "auroc": res_a4["auroc"],
        "auprc": res_a4["auprc"],
        "f1": res_a4["f1"],
        "far": res_a4["far"],
    }

    # A5: Reconstruction Score Only (w_rec=1.0, w_phys=0.0)
    logger.info("Running Ablation A5: Reconstruction Score Only...")
    proposed_base.w_rec = 1.0
    proposed_base.w_phys = 0.0
    c_s5, _, _, _ = proposed_base.score(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    t_s5, _, _, _ = proposed_base.score(X_test_win, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])
    cal_a5 = ConformalCalibrator(alpha_values=[0.05]).fit(c_s5)
    res_a5 = compute_detection_metrics(y_test_global, t_s5, threshold=cal_a5.thresholds[0.05])
    ablation_results["A5_Recon_Score_Only"] = {
        "description": "Scoring uses reconstruction error only (w_rec=1.0, w_phys=0.0)",
        "auroc": res_a5["auroc"],
        "auprc": res_a5["auprc"],
        "f1": res_a5["f1"],
        "far": res_a5["far"],
    }

    # A6: Physics Residual Score Only (w_rec=0.0, w_phys=1.0)
    logger.info("Running Ablation A6: Physics Residual Score Only...")
    proposed_base.w_rec = 0.0
    proposed_base.w_phys = 1.0
    c_s6, _, _, _ = proposed_base.score(X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3])
    t_s6, _, _, _ = proposed_base.score(X_test_win, Y_test_win, Y_test_raw[:, :, 0], Y_test_raw[:, :, 1], Y_test_raw[:, :, 2], Y_test_raw[:, :, 3])
    cal_a6 = ConformalCalibrator(alpha_values=[0.05]).fit(c_s6)
    res_a6 = compute_detection_metrics(y_test_global, t_s6, threshold=cal_a6.thresholds[0.05])
    ablation_results["A6_Physics_Score_Only"] = {
        "description": "Scoring uses physics residual only (w_rec=0.0, w_phys=1.0)",
        "auroc": res_a6["auroc"],
        "auprc": res_a6["auprc"],
        "f1": res_a6["f1"],
        "far": res_a6["far"],
    }

    # Reset proposed base weights
    proposed_base.w_rec = 0.5
    proposed_base.w_phys = 0.5

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(ablation_results, os.path.join(args.output_dir, "ablation_results.json"))
    logger.info("Ablation study completed and saved to: %s", os.path.join(args.output_dir, "ablation_results.json"))


if __name__ == "__main__":
    main()
