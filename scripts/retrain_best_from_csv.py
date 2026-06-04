#!/usr/bin/env python3
"""Re-train a single config from evolvegcn_experiments.csv (no grid search)."""
import argparse
import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import EXPERIMENTS_CSV, SNAPSHOTS_PATH
from src.evolvegcn.training import TrainConfig, run_training, run_drift_probe, build_summary_json, save_test_outputs
from src.evolvegcn.config import EVOLVE_SUMMARY_PATH, BEST_CHECKPOINT, PROCESSED_DIR
from src.evolvegcn.figures import generate_all_figures
from src.evolvegcn.config import FIGURES_DIR, STATIC_SUMMARY_PATH
import json


def select_from_csv(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    viable = [r for r in rows if float(r["val_auroc"]) > 0.51]
    if not viable:
        raise RuntimeError("No rows with val_auroc > 0.51 in CSV")
    return max(
        viable,
        key=lambda r: (
            float(r["val_auroc"]),
            float(r["post_t43_f1"]),
            float(r["test_auroc"]),
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None, help="e.g. r024_lr0.0001_cw5.0_h64_do0.3")
    p.add_argument("--csv", default=str(EXPERIMENTS_CSV))
    args = p.parse_args()

    with open(args.csv) as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}

    row = rows[args.run_id] if args.run_id else select_from_csv(Path(args.csv))
    print("Selected:", row["run_id"])

    cfg = TrainConfig(
        lr=float(row["lr"]),
        class_weight_illicit=float(row["class_weight"]),
        hidden1=int(row["hidden_dim"]),
        hidden2=int(row["hidden_dim"]) // 2,
        dropout=float(row["dropout"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)
    result = run_training(cfg, snapshots, device, verbose=True)
    torch.save(result["model"].state_dict(), BEST_CHECKPOINT)

    drift = run_drift_probe(cfg, snapshots, device)
    summary = build_summary_json(cfg, result, drift)
    EVOLVE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    save_test_outputs(result["model"], snapshots, device, PROCESSED_DIR / "evolvegcn_outputs.pt")

    with STATIC_SUMMARY_PATH.open() as f:
        static = json.load(f)
    sp = static["per_snapshot"]
    pre = sum(r["F1"] for r in sp if r["time_step"] < 43) / sum(1 for r in sp if r["time_step"] < 43)
    post = sum(r["F1"] for r in sp if r["time_step"] > 43) / sum(1 for r in sp if r["time_step"] > 43)
    generate_all_figures(
        result["history"],
        result["per_snapshot"],
        sp,
        result["pre_t43_f1"],
        result["post_t43_f1"],
        pre,
        post,
        FIGURES_DIR,
    )
    print("test_metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
