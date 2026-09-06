"""Robustness evaluation across measurement noise, packet loss, and operating shifts."""

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
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed
from gridguard.utils.io import load_parquet, load_json, save_json

logger = setup_logging()


def apply_measurement_noise(df: pd.DataFrame, noise_std: float, feature_cols: list, seed: int = 42) -> pd.DataFrame:
    """Add Gaussian measurement noise to telemetry features."""
    if noise_std <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    df_noisy = df.copy()
    for col in feature_cols:
        noise = rng.normal(0, noise_std, size=len(df_noisy))
        df_noisy[col] = df_noisy[col] + noise
    return df_noisy


def apply_causal_missingness(df: pd.DataFrame, missing_rate: float, feature_cols: list, seed: int = 42) -> pd.DataFrame:
    """Simulate telemetry packet drops with causal Last-Observation-Carried-Forward (LOCF) imputation."""
    if missing_rate <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    df_imp = df.sort_values(by=["node_id", "timestep"]).copy()

    # Apply packet drop mask
    for node in df_imp["node_id"].unique():
        node_mask = df_imp["node_id"] == node
        node_rows = df_imp[node_mask].index

        # Drop packets randomly
        drops = rng.uniform(0, 1, size=len(node_rows)) < missing_rate
        # Never drop first observation
        drops[0] = False

        # Set dropped to NaN and apply causal ffill
        for col in feature_cols:
            vals = df_imp.loc[node_rows, col].to_numpy(copy=True)
            vals[drops] = np.nan
            s = pd.Series(vals).ffill()
            df_imp.loc[node_rows, col] = s.to_numpy()

        df_imp.loc[node_rows[drops], "observation_mask"] = 0

    return df_imp.sort_values(by=["timestep", "node_id"])


def main():
    parser = argparse.ArgumentParser(description="Run robustness evaluation")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Dataset directory")
    parser.add_argument("--models_dir", type=str, default="artifacts/models", help="Models directory")
    parser.add_argument("--output_dir", type=str, default="artifacts/robustness", help="Output directory")
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

    _, df_calib, df_test = chronological_split(
        df,
        train_days=split_cfg["train"],
        calibration_days=split_cfg["calibration"],
        test_days=split_cfg["test"],
        steps_per_day=steps_per_day,
    )

    feat_cols = cfg.get("data", {}).get("features", ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"])
    scaler = FeatureScaler.from_dict(load_json(os.path.join(args.models_dir, "scaler.json")))

    win_len = cfg.get("data", {}).get("window_length", 8)

    # Clean calibration scores
    df_calib_scaled = scaler.transform_df(df_calib)
    X_cal_win, Y_cal_win, _, _, _ = create_sliding_windows(df_calib_scaled, window_length=win_len, feature_cols=feat_cols)
    X_cal_raw, Y_cal_raw, _, _, _ = create_sliding_windows(df_calib, window_length=win_len, feature_cols=feat_cols)

    with open(os.path.join(args.models_dir, "proposed_model.pkl"), "rb") as f:
        proposed_model = pickle.load(f)

    c_scores, _, _, _ = proposed_model.score(
        X_cal_win, Y_cal_win, Y_cal_raw[:, :, 0], Y_cal_raw[:, :, 1], Y_cal_raw[:, :, 2], Y_cal_raw[:, :, 3]
    )
    calibrator = ConformalCalibrator(alpha_values=[0.05]).fit(c_scores)
    thresh = calibrator.thresholds[0.05]

    noise_dict = {"none": 0.0, "low": 0.005, "moderate": 0.015}
    missing_dict = {"none": 0.0, "low": 0.01, "moderate": 0.05}

    matrix_f1 = np.zeros((len(noise_dict), len(missing_dict)))
    robustness_records = {}

    for i, (noise_name, noise_val) in enumerate(noise_dict.items()):
        for j, (miss_name, miss_val) in enumerate(missing_dict.items()):
            scenario_key = f"noise_{noise_name}_miss_{miss_name}"
            logger.info("Evaluating robustness scenario: %s (noise=%.3f, miss=%.2f)...", scenario_key, noise_val, miss_val)

            df_test_deg = apply_measurement_noise(df_test, noise_val, feat_cols, seed=seed)
            df_test_deg = apply_causal_missingness(df_test_deg, miss_val, feat_cols, seed=seed)

            df_test_scaled = scaler.transform_df(df_test_deg)
            X_t_win, Y_t_win, y_t_global, _, _ = create_sliding_windows(df_test_scaled, window_length=win_len, feature_cols=feat_cols)
            X_t_raw, Y_t_raw, _, _, _ = create_sliding_windows(df_test_deg, window_length=win_len, feature_cols=feat_cols)

            test_scores, _, _, _ = proposed_model.score(
                X_t_win, Y_t_win, Y_t_raw[:, :, 0], Y_t_raw[:, :, 1], Y_t_raw[:, :, 2], Y_t_raw[:, :, 3]
            )

            metrics = compute_detection_metrics(y_t_global, test_scores, threshold=thresh)
            matrix_f1[i, j] = metrics["f1"]
            robustness_records[scenario_key] = {
                "noise_level": noise_name,
                "noise_std": noise_val,
                "missing_rate_level": miss_name,
                "missing_rate": miss_val,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "far": metrics["far"],
            }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(robustness_records, os.path.join(args.output_dir, "robustness_matrix.json"))
    np.save(os.path.join(args.output_dir, "f1_matrix.npy"), matrix_f1)
    logger.info("Robustness evaluation complete and saved to: %s", os.path.join(args.output_dir, "robustness_matrix.json"))


if __name__ == "__main__":
    main()
