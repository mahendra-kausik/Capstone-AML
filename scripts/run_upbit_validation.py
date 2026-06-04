#!/usr/bin/env python3
"""
Full EthereumHeist (UpbitHack) validation pipeline.

Steps:
  1. Audit preprocessing artifacts
  2. Train Static GCN  → upbit_static_summary.json
  3. Train EvolveGCN-H → upbit_evolve_summary.json
  4. SHAP + Kendall τ
  5–6. Cross-dataset tables, figures, IEEE Results draft

Prerequisites:
  - data/processed_upbit/snapshots.pt (from Colab or scripts/preprocess_upbit.py)
  - Optional: UPBIT_PROCESSED_DIR env var for external path

Does NOT modify Elliptic artifacts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_script(name: str, extra: list[str] | None = None) -> None:
    cmd = [PY, str(REPO / "scripts" / name)] + (extra or [])
    print("\n" + "=" * 60)
    print(" ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, check=True, cwd=REPO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing if raw CSVs exist in data/raw/upbit/",
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--shap-force", action="store_true")
    p.add_argument("--steps", default="1,2,3,4,5", help="Comma-separated step numbers")
    args = p.parse_args()
    steps = {int(s.strip()) for s in args.steps.split(",")}

    if args.preprocess and 1 in steps:
        run_script("preprocess_upbit.py")

    if 1 in steps:
        try:
            run_script("audit_upbit_dataset.py")
        except subprocess.CalledProcessError as e:
            if e.returncode == 2:
                print(
                    "\nSTOP: snapshots.pt missing. Copy from Google Drive:\n"
                    "  Capstone/AML Code/data/processed_upbit/\n"
                    "or set UPBIT_PROCESSED_DIR and re-run.\n"
                    "Or download EthereumHeist CSVs and: python scripts/preprocess_upbit.py"
                )
                sys.exit(2)
            raise

    if 2 in steps and not args.skip_train:
        run_script("train_upbit_static.py")
    if 3 in steps and not args.skip_train:
        run_script("train_upbit_evolve.py")
    if 4 in steps and not args.skip_shap:
        extra = ["--force"] if args.shap_force else []
        run_script("run_upbit_shap.py", extra)
    if 5 in steps:
        run_script("generate_upbit_publication.py")

    print("\nUpbitHack validation pipeline complete.")


if __name__ == "__main__":
    main()
