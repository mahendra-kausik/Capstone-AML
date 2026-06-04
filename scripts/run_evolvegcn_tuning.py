#!/usr/bin/env python3
"""
EvolveGCN-H hyperparameter search, best-model selection, and artifact export.

Usage:
  python scripts/run_evolvegcn_tuning.py              # full grid (36 runs)
  python scripts/run_evolvegcn_tuning.py --quick      # smoke test (1 run)
  python scripts/run_evolvegcn_tuning.py --resume     # append to existing CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import (  # noqa: E402
    CLASS_WEIGHT_GRID,
    DROPOUT_GRID,
    EVOLVE_SUMMARY_PATH,
    EXPERIMENTS_CSV,
    BEST_CHECKPOINT,
    BEST_HISTORY_PATH,
    FIGURES_DIR,
    HIDDEN_DIM_GRID,
    LR_GRID,
    MAX_EPOCHS,
    MODELS_DIR,
    PATIENCE,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    STATIC_SUMMARY_PATH,
    WEIGHT_DECAY,
    SNAPSHOTS_PATH,
)
from src.evolvegcn.figures import generate_all_figures  # noqa: E402
from src.evolvegcn.training import (  # noqa: E402
    TrainConfig,
    build_summary_json,
    get_device,
    pre_post_t43_f1,
    run_drift_probe,
    run_training,
    save_test_outputs,
)


CSV_FIELDS = [
    "run_id",
    "lr",
    "class_weight",
    "hidden_dim",
    "dropout",
    "val_auroc",
    "test_auroc",
    "test_f1",
    "pre_t43_f1",
    "post_t43_f1",
]


def iter_grid(quick: bool):
    grid = list(
        product(LR_GRID, CLASS_WEIGHT_GRID, HIDDEN_DIM_GRID, DROPOUT_GRID)
    )
    if quick:
        return [grid[0]]
    return grid


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def select_best(rows: list[dict]) -> dict:
    """
    Priority: val_auroc (must be informative) → post_t43_f1 → test_auroc → test_f1.
    Reject collapsed runs (val_auroc <= 0.51) so AUROC=0.5 tie-break junk is not chosen.
    """
    viable = [r for r in rows if float(r["val_auroc"]) > 0.51]
    pool = viable if viable else rows

    def key(r):
        return (
            float(r["val_auroc"]),
            float(r["post_t43_f1"]),
            float(r["test_auroc"]),
            float(r["test_f1"]),
        )

    return max(pool, key=key)


def static_pre_post() -> tuple[float, float, list]:
    with STATIC_SUMMARY_PATH.open() as f:
        s = json.load(f)
    per = s["per_snapshot"]
    pre = [r["F1"] for r in per if r["time_step"] < 43]
    post = [r["F1"] for r in per if r["time_step"] > 43]
    return float(sum(pre) / len(pre)), float(sum(post) / len(post)), per


def main():
    parser = argparse.ArgumentParser(description="EvolveGCN-H tuning pipeline")
    parser.add_argument("--quick", action="store_true", help="Run a single config smoke test")
    parser.add_argument(
        "--quick-epochs",
        type=int,
        default=3,
        help="Max epochs per run when --quick (default 3)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip run_ids already in CSV")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)

    existing = load_existing_ids(EXPERIMENTS_CSV) if args.resume else set()
    rows: list[dict] = []

    if EXPERIMENTS_CSV.exists() and args.resume:
        with EXPERIMENTS_CSV.open() as f:
            rows = list(csv.DictReader(f))

    grid = iter_grid(args.quick)
    epoch_cap = args.quick_epochs if args.quick else MAX_EPOCHS
    patience_cap = min(PATIENCE, max(1, epoch_cap - 1)) if args.quick else PATIENCE
    print(f"Planned runs: {len(grid)}  (max_epochs={epoch_cap})")

    for run_idx, (lr, cw, hidden, dropout) in enumerate(grid):
        run_id = f"r{run_idx:03d}_lr{lr}_cw{cw}_h{hidden}_do{dropout}"
        if run_id in existing:
            print(f"Skip {run_id} (already logged)")
            continue

        cfg = TrainConfig(
            lr=lr,
            class_weight_illicit=cw,
            hidden1=hidden,
            hidden2=hidden // 2,
            dropout=dropout,
            weight_decay=WEIGHT_DECAY,
            max_epochs=epoch_cap,
            patience=patience_cap,
            seed=args.seed,
        )
        print(f"\n=== {run_id} ===")
        print(f"  config: {cfg}")

        result = run_training(cfg, snapshots, device, verbose=True)
        row = {
            "run_id": run_id,
            "lr": lr,
            "class_weight": cw,
            "hidden_dim": hidden,
            "dropout": dropout,
            "val_auroc": round(result["best_val_auroc"], 4),
            "test_auroc": round(result["test_metrics"]["AUROC"], 4),
            "test_f1": round(result["test_metrics"]["F1"], 4),
            "pre_t43_f1": round(result["pre_t43_f1"], 4),
            "post_t43_f1": round(result["post_t43_f1"], 4),
        }
        rows.append(row)

        with EXPERIMENTS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  logged → {EXPERIMENTS_CSV}")

    if not rows:
        print("No experiment rows available.")
        return

    best_row = select_best(rows)
    print(f"\nBest run (post_t43_f1 → test_auroc → test_f1): {best_row['run_id']}")

    final_epochs = epoch_cap if args.quick else MAX_EPOCHS
    final_patience = patience_cap if args.quick else PATIENCE
    best_cfg = TrainConfig(
        lr=float(best_row["lr"]),
        class_weight_illicit=float(best_row["class_weight"]),
        hidden1=int(best_row["hidden_dim"]),
        hidden2=int(best_row["hidden_dim"]) // 2,
        dropout=float(best_row["dropout"]),
        weight_decay=WEIGHT_DECAY,
        max_epochs=final_epochs,
        patience=final_patience,
        seed=args.seed,
    )
    print(f"Re-training best config: {best_cfg}")
    best_result = run_training(best_cfg, snapshots, device, verbose=True)

    torch.save(best_result["model"].state_dict(), BEST_CHECKPOINT)
    print(f"Checkpoint → {BEST_CHECKPOINT}")

    with BEST_HISTORY_PATH.open("w") as f:
        json.dump(best_result["history"], f, indent=2)

    drift = run_drift_probe(best_cfg, snapshots, device)
    summary = build_summary_json(best_cfg, best_result, drift)
    with EVOLVE_SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {EVOLVE_SUMMARY_PATH}")

    save_test_outputs(
        best_result["model"],
        snapshots,
        device,
        PROCESSED_DIR / "evolvegcn_outputs.pt",
    )

    s_pre, s_post, static_per = static_pre_post()
    generate_all_figures(
        best_result["history"],
        best_result["per_snapshot"],
        static_per,
        best_result["pre_t43_f1"],
        best_result["post_t43_f1"],
        s_pre,
        s_post,
    )
    print(f"Figures → {FIGURES_DIR}")

    print("\n=== Best model metrics ===")
    for k, v in best_result["test_metrics"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  pre_t43_f1:  {best_result['pre_t43_f1']:.4f}")
    print(f"  post_t43_f1: {best_result['post_t43_f1']:.4f}")


if __name__ == "__main__":
    main()
