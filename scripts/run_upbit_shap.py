#!/usr/bin/env python3
"""Step 4: SHAP + Kendall τ drift on UpbitHack."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.training import get_device
from src.upbit.config import EVOLVE_CHECKPOINT, SNAPSHOTS_PATH, STATIC_CHECKPOINT
from src.upbit.shap_analysis import run_upbit_shap_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if not SNAPSHOTS_PATH.exists():
        print(f"Missing {SNAPSHOTS_PATH}")
        sys.exit(2)
    for ckpt in (STATIC_CHECKPOINT, EVOLVE_CHECKPOINT):
        if not ckpt.exists():
            print(f"Missing checkpoint {ckpt}. Train models first.")
            sys.exit(2)

    device = get_device() if args.device is None else __import__("torch").device(args.device)
    out = run_upbit_shap_pipeline(
        static_ckpt=STATIC_CHECKPOINT,
        evolve_ckpt=EVOLVE_CHECKPOINT,
        device=device,
        force=args.force,
    )

    print("\nKendall τ (Upbit Static):")
    for r in out["static_tau"]:
        print(f"  {r['comparison']}: {r['tau']:.4f}")
    print("\nKendall τ (Upbit Evolve):")
    for r in out["evolve_tau"]:
        print(f"  {r['comparison']}: {r['tau']:.4f}")


if __name__ == "__main__":
    main()
