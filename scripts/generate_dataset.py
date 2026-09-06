"""Dataset generation pipeline: OpenDSS simulation, load/PV profiles, and attack injection."""

import argparse
import os
import sys
import pandas as pd
import numpy as np

from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.feeder import FeederModel
from gridguard.simulation.measurements import extract_measurements
from gridguard.simulation.profiles import LoadProfileGenerator
from gridguard.simulation.pv import PVProfileGenerator
from gridguard.simulation.ybus import YBusModel
from gridguard.attacks.scheduler import AttackScheduler
from gridguard.data.schema import validate_dataframe_schema
from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.reproducibility import set_seed, calculate_file_hash, get_environment_info
from gridguard.utils.io import save_parquet, save_json

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic power-grid telemetry dataset")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    sim_cfg = cfg.get("simulation", {})
    feeder_file = sim_cfg.get("feeder_file", "tests/fixtures/tiny_feeder.dss")
    num_days = sim_cfg.get("days", 5)
    res_min = sim_cfg.get("resolution_minutes", 15)
    base_mva = cfg.get("data", {}).get("base_mva", 1.0)
    steps_per_day = (24 * 60) // res_min
    total_timesteps = num_days * steps_per_day

    logger.info("Initializing OpenDSS circuit: %s", feeder_file)
    dss_iface = OpenDSSInterface(feeder_file)
    feeder_model = FeederModel(dss_iface)
    ybus_model = YBusModel(dss_iface, base_mva=base_mva)

    load_names = dss_iface.get_load_names()
    nominal_loads = dss_iface.get_load_data()
    num_nodes = feeder_model.num_nodes
    logger.info("Feeder loaded: %d nodes, %d loads, simulating %d days (%d steps)", num_nodes, len(load_names), num_days, total_timesteps)

    # 1. Generate stochastic load profiles
    load_gen = LoadProfileGenerator(
        load_names=load_names,
        resolution_minutes=res_min,
        seed=seed,
    )
    load_profiles = load_gen.generate_profiles(num_days=num_days)

    # 2. Run simulation and extract measurements
    records = []
    logger.info("Running time-series power flow simulations...")

    for t in range(total_timesteps):
        # Set individual loads
        for l_idx, l_name in enumerate(load_names):
            if l_name in nominal_loads:
                base_kw = nominal_loads[l_name]["kw"]
                base_kvar = nominal_loads[l_name]["kvar"]
                mult = load_profiles[t, l_idx]
                dss_iface.set_individual_load(l_name, kw=base_kw * mult, kvar=base_kvar * mult)

        converged = dss_iface.solve()
        if not converged:
            logger.warning("Simulation step %d failed to converge!", t)

        meas = extract_measurements(dss_iface, feeder_model, base_mva=base_mva)

        for n_idx, meta in enumerate(feeder_model.node_metadata):
            record = {
                "timestamp": f"Day_{t // steps_per_day + 1}_Step_{t % steps_per_day:02d}",
                "timestep": int(t),
                "scenario": cfg.get("experiment", {}).get("name", "standard"),
                "node_id": int(meta["node_id"]),
                "bus": str(meta["bus_name"]),
                "phase": int(meta["phase_num"]),
                "v_mag_pu": float(meas["v_mag_pu"][n_idx]),
                "v_ang_rad": float(meas["v_ang_rad"][n_idx]),
                "sin_theta": float(meas["sin_theta"][n_idx]),
                "cos_theta": float(meas["cos_theta"][n_idx]),
                "p_inj_pu": float(meas["p_inj_pu"][n_idx]),
                "q_inj_pu": float(meas["q_inj_pu"][n_idx]),
                # Clean ground truth copies
                "clean_v_mag_pu": float(meas["v_mag_pu"][n_idx]),
                "clean_v_ang_rad": float(meas["v_ang_rad"][n_idx]),
                "clean_p_inj_pu": float(meas["p_inj_pu"][n_idx]),
                "clean_q_inj_pu": float(meas["q_inj_pu"][n_idx]),
                "observation_mask": int(1),
                "is_attack": int(0),
                "attack_id": "none",
                "attack_type": "none",
                "attack_strength": float(0.0),
                "is_compromised": int(0),
            }
            records.append(record)

    df_clean = pd.DataFrame(records)
    logger.info("Generated %d total telemetry records.", len(df_clean))

    # 3. Schedule cyberattacks on test split
    split_days = sim_cfg.get("split_days", {"train": 3, "calibration": 1, "test": 1})
    train_end_step = split_days.get("train", 3) * steps_per_day
    calib_end_step = train_end_step + split_days.get("calibration", 1) * steps_per_day

    df_train_calib = df_clean[df_clean["timestep"] < calib_end_step].copy()
    df_test_clean = df_clean[df_clean["timestep"] >= calib_end_step].copy()

    logger.info("Scheduling non-overlapping cyberattacks on test split (steps >= %d)...", calib_end_step)
    scheduler = AttackScheduler(attack_config=cfg.get("attacks", {}), seed=seed)

    extra_context = {
        "graph": feeder_model.graph,
        "node_names": feeder_model.node_names,
        "dss_interface": dss_iface,
        "feeder_model": feeder_model,
        "load_profiles": load_profiles,
        "nominal_loads": nominal_loads,
        "base_mva": base_mva,
    }

    df_test_attacked, attack_metadata_list = scheduler.schedule_attacks(
        df_test=df_test_clean,
        num_nodes=num_nodes,
        extra_context=extra_context,
    )

    df_final = pd.concat([df_train_calib, df_test_attacked], ignore_index=True)
    validate_dataframe_schema(df_final)

    # 4. Save dataset and provenance artifacts
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(out_dir, "telemetry_dataset.parquet")
    save_parquet(df_final, dataset_path)

    # Save graph adjacency and node metadata
    adj_matrix = feeder_model.get_adjacency_matrix()
    np.save(os.path.join(out_dir, "adjacency_matrix.npy"), adj_matrix)
    save_json(feeder_model.node_metadata, os.path.join(out_dir, "node_metadata.json"))

    # Save attack metadata and environment provenance
    save_json([m.__dict__ for m in attack_metadata_list], os.path.join(out_dir, "attack_metadata.json"))
    env_info = get_environment_info(seed=seed)
    env_info["dataset_hash"] = calculate_file_hash(dataset_path)
    save_json(env_info, os.path.join(out_dir, "provenance.json"))

    logger.info("Dataset generated and verified successfully at: %s", dataset_path)


if __name__ == "__main__":
    main()
