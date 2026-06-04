#!/usr/bin/env python3
"""Step 3: Train patched EvolveGCN-H on UpbitHack."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.training import (
    TrainConfig,
    build_summary_json,
    get_device,
    is_collapsed_predictions,
    run_drift_probe,
    run_training,
)
from src.upbit.config import (
    EVOLVE_CHECKPOINT,
    EVOLVE_CLASS_WEIGHT,
    EVOLVE_DROPOUT,
    EVOLVE_LR,
    EVOLVE_MAX_EPOCHS,
    EVOLVE_PATIENCE,
    EVOLVE_SUMMARY_PATH,
    SEED,
    SNAPSHOTS_PATH,
)


def main():
    if not SNAPSHOTS_PATH.exists():
        print(f"Missing {SNAPSHOTS_PATH}")
        sys.exit(2)

    device = get_device()
    print(f"Device: {device}")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)

    cfg = TrainConfig(
        lr=EVOLVE_LR,
        class_weight_illicit=EVOLVE_CLASS_WEIGHT,
        hidden1=64,
        hidden2=32,
        dropout=EVOLVE_DROPOUT,
        max_epochs=EVOLVE_MAX_EPOCHS,
        patience=EVOLVE_PATIENCE,
        seed=SEED,
    )

    print("Training EvolveGCN-H on UpbitHack (patched)...")
    result = run_training(cfg, snapshots, device, verbose=True)
    drift = run_drift_probe(cfg, snapshots, device)

    from src.evolvegcn.config import TEST_IDX
    from src.evolvegcn.training import _forward_probs

    _, y_prob, logits = _forward_probs(
        result["model"], snapshots, TEST_IDX, device, reset_first=True
    )
    if is_collapsed_predictions(y_prob, logits):
        print("WARNING: collapsed predictions on test set")

    summary = build_summary_json(cfg, result, drift)
    summary["dataset"] = "UpbitHack"
    summary["model_version"] = "evolvegcn-h-patched-v1"

    EVOLVE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVOLVE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    EVOLVE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": result["model"].state_dict(), "config": cfg.__dict__},
        EVOLVE_CHECKPOINT,
    )

    tm = summary["test_metrics"]
    print("\n=== Test metrics ===")
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved {EVOLVE_SUMMARY_PATH}")
    print(f"Saved {EVOLVE_CHECKPOINT}")


if __name__ == "__main__":
    main()
