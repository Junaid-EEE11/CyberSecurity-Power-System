"""Generate publication tables (Tables 1 to 8) in CSV and Markdown/LaTeX formats."""

import argparse
import os
import pandas as pd
import numpy as np

from gridguard.utils.config import load_config
from gridguard.utils.logging import setup_logging
from gridguard.utils.io import load_json

logger = setup_logging()


def save_table(df: pd.DataFrame, table_name: str, output_dir: str) -> None:
    """Save table as CSV and Markdown formatted files."""
    csv_path = os.path.join(output_dir, f"{table_name}.csv")
    md_path = os.path.join(output_dir, f"{table_name}.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {table_name.replace('_', ' ').title()}\n\n")
        f.write(df.to_markdown(index=False))
    logger.info("Saved %s (CSV & Markdown)", table_name)


def main():
    parser = argparse.ArgumentParser(description="Generate publication tables")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config")
    parser.add_argument("--eval_dir", type=str, default="artifacts/evaluation", help="Evaluation directory")
    parser.add_argument("--ablation_dir", type=str, default="artifacts/ablations", help="Ablation directory")
    parser.add_argument("--robustness_dir", type=str, default="artifacts/robustness", help="Robustness directory")
    parser.add_argument("--output_dir", type=str, default="artifacts/tables", help="Tables output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Table 1: Dataset and Simulation Configuration
    sim = cfg.get("simulation", {})
    t1_data = [
        {"Parameter": "Primary Test Feeder", "Value": sim.get("feeder_type", "IEEE 123-Bus / Synthetic Fixture")},
        {"Parameter": "Simulation Resolution", "Value": f"{sim.get('resolution_minutes', 15)} minutes (96 steps/day)"},
        {"Parameter": "Total Operating Days", "Value": f"{sim.get('days', 180)} days"},
        {"Parameter": "Train Split (Clean Normal Only)", "Value": f"{sim.get('split_days', {}).get('train', 120)} days"},
        {"Parameter": "Calibration Split (Clean Normal Only)", "Value": f"{sim.get('split_days', {}).get('calibration', 30)} days"},
        {"Parameter": "Test Split (Clean & Attacked)", "Value": f"{sim.get('split_days', {}).get('test', 30)} days"},
        {"Parameter": "Sliding Window Length", "Value": f"{cfg.get('data', {}).get('window_length', 12)} timesteps (3.0 hours)"},
        {"Parameter": "Telemetry Feature Channels", "Value": "V_mag (p.u.), V_ang (rad), P_inj (p.u.), Q_inj (p.u.)"},
    ]
    save_table(pd.DataFrame(t1_data), "table_01_simulation_config", args.output_dir)

    # Table 2: Attack Definitions
    t2_data = [
        {"Attack ID": "Attack A", "Family": "Step FDIA", "Mechanism": "Constant relative injection bias", "Target Nodes": "5% random bus-phases", "Strengths": "0.03, 0.08, 0.15"},
        {"Attack ID": "Attack B", "Family": "Ramp FDIA", "Mechanism": "Linear slow drift over duration", "Target Nodes": "5% random bus-phases", "Strengths": "0.03, 0.08, 0.15"},
        {"Attack ID": "Attack C", "Family": "Replay FDIA", "Mechanism": "Substitution of stale historical telemetry", "Target Nodes": "5% random bus-phases", "Strengths": "24 to 48 lag steps"},
        {"Attack ID": "Attack D", "Family": "Coordinated Spatial", "Mechanism": "Simultaneous perturbation of connected subgraph", "Target Nodes": "2-hop topological cluster", "Strengths": "0.08, 0.15"},
        {"Attack ID": "Attack E", "Family": "Operating-Point Substitution (Unseen)", "Mechanism": "Converged power flow state from perturbed loads", "Target Nodes": "8% load bus-phases", "Strengths": "+/-20% to 30% load shift"},
    ]
    save_table(pd.DataFrame(t2_data), "table_02_attack_definitions", args.output_dir)

    # Table 3: Overall Detection Results
    eval_summary_path = os.path.join(args.eval_dir, "metrics_summary.json")
    if os.path.exists(eval_summary_path):
        m_sum = load_json(eval_summary_path)
        t3_rows = []
        for model_name, m_data in m_sum.items():
            ov = m_data["overall"]
            del_m = m_data.get("delay", {})
            t3_rows.append({
                "Model": model_name,
                "AUROC": f"{ov['auroc']:.4f}",
                "AUPRC": f"{ov['auprc']:.4f}",
                "Precision": f"{ov['precision']:.4f}",
                "Recall": f"{ov['recall']:.4f}",
                "F1-Score": f"{ov['f1']:.4f}",
                "FAR (Empirical)": f"{ov['far']:.4f}",
                "Recall @ 5% FAR": f"{ov['recall_at_05_far']:.4f}",
                "Mean Delay (min)": f"{del_m.get('mean_delay_minutes', 0.0):.1f}",
            })
        save_table(pd.DataFrame(t3_rows), "table_03_overall_detection_results", args.output_dir)

        # Table 4: Per-Attack Performance
        t4_rows = []
        for model_name, m_data in m_sum.items():
            per_att = m_data.get("per_attack", {})
            for att_name, att_metrics in per_att.items():
                t4_rows.append({
                    "Model": model_name,
                    "Attack Type": att_name,
                    "AUROC": f"{att_metrics['auroc']:.4f}",
                    "AUPRC": f"{att_metrics['auprc']:.4f}",
                    "Recall": f"{att_metrics['recall']:.4f}",
                    "F1-Score": f"{att_metrics['f1']:.4f}",
                })
        if t4_rows:
            save_table(pd.DataFrame(t4_rows), "table_04_per_attack_performance", args.output_dir)

        # Table 5: Unseen Attack E Results
        t5_rows = []
        for model_name, m_data in m_sum.items():
            att_e = m_data.get("per_attack", {}).get("Attack_E_OperatingPoint", {})
            if att_e:
                t5_rows.append({
                    "Model": model_name,
                    "Unseen Attack E AUROC": f"{att_e['auroc']:.4f}",
                    "Unseen Attack E AUPRC": f"{att_e['auprc']:.4f}",
                    "Unseen Attack E Recall": f"{att_e['recall']:.4f}",
                    "Unseen Attack E F1": f"{att_e['f1']:.4f}",
                })
        if t5_rows:
            save_table(pd.DataFrame(t5_rows), "table_05_unseen_attack_e_results", args.output_dir)

        # Table 7: Localization Results
        t7_rows = []
        for model_name, m_data in m_sum.items():
            loc = m_data["localization"]
            t7_rows.append({
                "Model": model_name,
                "Top-1 Hit Rate": f"{loc['top1_hit']:.4f}",
                "Top-3 Hit Rate": f"{loc['top3_hit']:.4f}",
                "Top-5 Hit Rate": f"{loc['top5_hit']:.4f}",
                "Node Precision": f"{loc['node_precision']:.4f}",
                "Node Recall": f"{loc['node_recall']:.4f}",
                "Node F1": f"{loc['node_f1']:.4f}",
            })
        save_table(pd.DataFrame(t7_rows), "table_07_localization_results", args.output_dir)

    # Table 6: Robustness Matrix Results
    rob_path = os.path.join(args.robustness_dir, "robustness_matrix.json")
    if os.path.exists(rob_path):
        rob_dict = load_json(rob_path)
        t6_rows = []
        for scen, r in rob_dict.items():
            t6_rows.append({
                "Scenario": scen,
                "Noise Level": r["noise_level"],
                "Missing Rate": f"{r['missing_rate']*100:.1f}%",
                "AUROC": f"{r['auroc']:.4f}",
                "AUPRC": f"{r['auprc']:.4f}",
                "F1-Score": f"{r['f1']:.4f}",
                "Empirical FAR": f"{r['far']:.4f}",
            })
        save_table(pd.DataFrame(t6_rows), "table_06_robustness_results", args.output_dir)

    # Table 8: Ablation Results
    abl_path = os.path.join(args.ablation_dir, "ablation_results.json")
    if os.path.exists(abl_path):
        abl_dict = load_json(abl_path)
        t8_rows = []
        for v_name, v_data in abl_dict.items():
            t8_rows.append({
                "Variant": v_name,
                "Description": v_data["description"],
                "AUROC": f"{v_data['auroc']:.4f}",
                "AUPRC": f"{v_data['auprc']:.4f}",
                "F1-Score": f"{v_data['f1']:.4f}",
                "Empirical FAR": f"{v_data['far']:.4f}",
            })
        save_table(pd.DataFrame(t8_rows), "table_08_ablation_results", args.output_dir)

    logger.info("All publication tables generated successfully.")


if __name__ == "__main__":
    main()
