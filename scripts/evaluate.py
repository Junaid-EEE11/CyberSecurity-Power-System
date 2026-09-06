"""Comprehensive benchmark evaluation of all detectors and calibration analysis."""

import argparse
import os
import pickle
from typing import Dict, List
import numpy as np
import pandas as pd

from gridguard.calibration.conformal import ConformalCalibrator
from gridguard.data.splits import chronological_split
from gridguard.data.scaling import FeatureScaler
from gridguard.data.windows import create_sliding_windows
from gridguard.evaluation.detection import compute_detection_metrics, compute_recall_at_far
from gridguard.evaluation.localization import compute_localization_metrics
from gridguard.evaluation.delay import compute_detection_delay
from gridguard.evaluation.statistics import bootstrap_confidence_interval, paired_wilcoxon_test
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed
from gridguard.utils.io import load_parquet, load_json, save_json

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detectors")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing dataset")
    parser.add_argument("--models_dir", type=str, default="artifacts/models", help="Directory containing trained models")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation", help="Evaluation output directory")
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

    df_calib_scaled = scaler.transform_df(df_calib)
    df_test_scaled = scaler.transform_df(df_test)

    win_len = cfg.get("data", {}).get("window_length", 8)

    # Calibration windows (all clean)
    X_cal_win, Y_cal_win, _, _, cal_steps = create_sliding_windows(df_calib_scaled, window_length=win_len, feature_cols=feat_cols)
    X_cal_raw, Y_cal_raw, _, _, _ = create_sliding_windows(df_calib, window_length=win_len, feature_cols=feat_cols)

    # Test windows (contains clean + attacked periods)
    X_test_win, Y_test_win, y_test_global, y_test_nodes, test_steps = create_sliding_windows(df_test_scaled, window_length=win_len, feature_cols=feat_cols)
    X_test_raw, Y_test_raw, _, _, _ = create_sliding_windows(df_test, window_length=win_len, feature_cols=feat_cols)

    # Load baseline models
    with open(os.path.join(args.models_dir, "baselines.pkl"), "rb") as f:
        all_models = pickle.load(f)

    # Load proposed model
    with open(os.path.join(args.models_dir, "proposed_model.pkl"), "rb") as f:
        proposed_model = pickle.load(f)
    all_models[proposed_model.name] = proposed_model

    logger.info("Loaded %d models for evaluation: %s", len(all_models), list(all_models.keys()))

    evaluation_results = {}
    model_score_arrays = {}

    for model_name, model in all_models.items():
        logger.info("Evaluating %s...", model_name)

        # 1. Compute calibration scores strictly on clean calibration period
        if model_name in ["B0_Statistical_MAD", "B1_PCA_SPE", "B2_IsolationForest", "B3_Dense_AE"]:
            calib_global, _ = model.score(Y_cal_win)
            test_global, test_nodes = model.score(Y_test_win)
        elif model_name in ["B4_LSTM_AE", "B6_GraphTemporal_AE"]:
            calib_global, _ = model.score(X_cal_win, Y_cal_win)
            test_global, test_nodes = model.score(X_test_win, Y_test_win)
        elif model_name == "B5_PhysicsResidualOnly":
            calib_global, _ = model.score(
                V_mag=Y_cal_raw[:, :, 0],
                V_ang_rad=Y_cal_raw[:, :, 1],
                P_inj_pu=Y_cal_raw[:, :, 2],
                Q_inj_pu=Y_cal_raw[:, :, 3],
            )
            test_global, test_nodes = model.score(
                V_mag=Y_test_raw[:, :, 0],
                V_ang_rad=Y_test_raw[:, :, 1],
                P_inj_pu=Y_test_raw[:, :, 2],
                Q_inj_pu=Y_test_raw[:, :, 3],
            )
        elif "Proposed" in model_name:
            calib_global, _, _, _ = model.score(
                X_windows=X_cal_win,
                Y_targets=Y_cal_win,
                raw_V_mag=Y_cal_raw[:, :, 0],
                raw_V_ang=Y_cal_raw[:, :, 1],
                raw_P_inj=Y_cal_raw[:, :, 2],
                raw_Q_inj=Y_cal_raw[:, :, 3],
            )
            test_global, test_nodes, _, _ = model.score(
                X_windows=X_test_win,
                Y_targets=Y_test_win,
                raw_V_mag=Y_test_raw[:, :, 0],
                raw_V_ang=Y_test_raw[:, :, 1],
                raw_P_inj=Y_test_raw[:, :, 2],
                raw_Q_inj=Y_test_raw[:, :, 3],
            )

        model_score_arrays[model_name] = {
            "scores": test_global,
            "node_scores": test_nodes,
        }

        # 2. Conformal Calibration on Clean Calibration Scores
        calibrator = ConformalCalibrator(alpha_values=[0.01, 0.05])
        calibrator.fit(calib_global)
        thresh_05 = calibrator.thresholds[0.05]
        thresh_01 = calibrator.thresholds[0.01]

        # 3. Overall Detection Metrics
        det_metrics = compute_detection_metrics(y_test_global, test_global, threshold=thresh_05)
        recall_at_05 = compute_recall_at_far(y_test_global, test_global, target_far=0.05)
        recall_at_01 = compute_recall_at_far(y_test_global, test_global, target_far=0.01)

        # 4. Localization Metrics
        loc_metrics = compute_localization_metrics(y_test_nodes, test_nodes, y_test_global)

        # 5. Detection Delay
        delay_metrics = compute_detection_delay(y_test_global, test_global, threshold=thresh_05, timesteps=test_steps)

        # 6. Clean Empirical False-Alarm Rate on Test Normal Samples
        clean_mask = (y_test_global == 0)
        clean_test_scores = test_global[clean_mask]
        calib_eval = calibrator.evaluate_clean_far(clean_test_scores)

        # 7. Per-Attack Family Breakdown
        attack_subsets = {}
        # Map test steps to attack types
        df_test_step_map = df_test.drop_duplicates(subset=["timestep"]).set_index("timestep")["attack_type"].to_dict()
        test_att_types = np.array([df_test_step_map.get(t, "none") for t in test_steps])

        for att_type in ["Attack_A_Step", "Attack_B_Ramp", "Attack_C_Replay", "Attack_D_Coordinated", "Attack_E_OperatingPoint"]:
            att_mask = (test_att_types == att_type)
            if att_mask.any():
                # Compare this attack subset against all normal samples
                eval_mask = att_mask | clean_mask
                sub_y = y_test_global[eval_mask]
                sub_scores = test_global[eval_mask]
                sub_metrics = compute_detection_metrics(sub_y, sub_scores, threshold=thresh_05)
                attack_subsets[att_type] = {
                    "auroc": sub_metrics["auroc"],
                    "auprc": sub_metrics["auprc"],
                    "recall": sub_metrics["recall"],
                    "f1": sub_metrics["f1"],
                }

        # Bootstrap CI for AUPRC
        _, auprc_ci_low, auprc_ci_high = bootstrap_confidence_interval(
            test_global[y_test_global == 1], n_bootstraps=200, seed=seed
        )

        evaluation_results[model_name] = {
            "overall": {
                "auroc": det_metrics["auroc"],
                "auprc": det_metrics["auprc"],
                "precision": det_metrics["precision"],
                "recall": det_metrics["recall"],
                "f1": det_metrics["f1"],
                "far": det_metrics["far"],
                "recall_at_05_far": recall_at_05,
                "recall_at_01_far": recall_at_01,
                "threshold_alpha_05": thresh_05,
                "threshold_alpha_01": thresh_01,
            },
            "localization": loc_metrics,
            "delay": delay_metrics,
            "calibration": calib_eval,
            "per_attack": attack_subsets,
        }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(evaluation_results, os.path.join(args.output_dir, "metrics_summary.json"))

    # Save score arrays for plotting
    score_save_path = os.path.join(args.output_dir, "model_scores.pkl")
    with open(score_save_path, "wb") as f:
        pickle.dump({"scores": model_score_arrays, "y_true": y_test_global, "test_steps": test_steps}, f)

    logger.info("Evaluation complete. Summary metrics saved to: %s", os.path.join(args.output_dir, "metrics_summary.json"))


if __name__ == "__main__":
    main()
