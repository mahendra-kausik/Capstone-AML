#!/usr/bin/env python3
"""Step 2: Train Static GCN on UpbitHack (same architecture as Elliptic)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.training import get_device
from src.static_gcn.training import (
    StaticTrainConfig,
    build_static_summary,
    run_static_drift_probe,
    run_static_training,
    save_summary,
)
from src.upbit.config import (
    SEED,
    SNAPSHOTS_PATH,
    STATIC_CHECKPOINT,
    STATIC_CLASS_WEIGHT,
    STATIC_DROPOUT,
    STATIC_HIDDEN1,
    STATIC_HIDDEN2,
    STATIC_LR,
    STATIC_MAX_EPOCHS,
    STATIC_PATIENCE,
    STATIC_SUMMARY_PATH,
    STATIC_WEIGHT_DECAY,
)


def main():
    if not SNAPSHOTS_PATH.exists():
        print(f"Missing {SNAPSHOTS_PATH}. Run audit_upbit_dataset.py first.")
        sys.exit(2)

    device = get_device()
    print(f"Device: {device}")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)

    cfg = StaticTrainConfig(
        lr=STATIC_LR,
        weight_decay=STATIC_WEIGHT_DECAY,
        max_epochs=STATIC_MAX_EPOCHS,
        patience=STATIC_PATIENCE,
        class_weight_illicit=STATIC_CLASS_WEIGHT,
        hidden1=STATIC_HIDDEN1,
        hidden2=STATIC_HIDDEN2,
        dropout=STATIC_DROPOUT,
        seed=SEED,
    )

    print("Training Static GCN on UpbitHack...")
    result = run_static_training(
        cfg, snapshots, device, STATIC_CHECKPOINT, verbose=True
    )
    drift = run_static_drift_probe(cfg, snapshots, device)
    summary = build_static_summary(cfg, result, drift)

    save_summary(STATIC_SUMMARY_PATH, summary)
    tm = summary["test_metrics"]
    print("\n=== Test metrics ===")
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved {STATIC_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
