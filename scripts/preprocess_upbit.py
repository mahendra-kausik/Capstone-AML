#!/usr/bin/env python3
"""Run UpbitHack preprocessing (from Notebooks/01_UpbitHack_Preprocessing.ipynb)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BODY = Path(__file__).with_name("_upbit_preprocess_body.py")


def main():
    p = argparse.ArgumentParser(description="Build data/processed_upbit from raw CSVs")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO / "data" / "raw" / "upbit",
        help="Directory with all-tx.csv, all-address.csv, accounts-hacker.csv",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO / "data" / "processed_upbit",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO / "figures" / "upbit",
    )
    args = p.parse_args()

    for f in ["all-tx.csv", "all-address.csv", "accounts-hacker.csv"]:
        if not (args.raw_dir / f).exists():
            print(f"Missing {args.raw_dir / f}")
            print(
                "Download UpbitHack from EthereumHeist Dropbox and place CSVs in --raw-dir"
            )
            sys.exit(1)

    if not BODY.exists():
        print(f"Missing {BODY}; re-extract from notebook.")
        sys.exit(1)

    text = BODY.read_text()
    text = text.replace(
        'RAW_DIR = Path("/content/sample_data")',
        f'RAW_DIR = Path({args.raw_dir!r})',
    )
    text = text.replace(
        'PROCESSED_DIR = Path("/content/drive/MyDrive/Capstone/AML Code/data/processed_upbit")',
        f'PROCESSED_DIR = Path({args.processed_dir!r})',
    )
    text = text.replace(
        'FIGURES_DIR   = Path("/content/drive/MyDrive/Capstone/AML Code/figures/upbit")',
        f'FIGURES_DIR = Path({args.figures_dir!r})',
    )
    # Drop notebook cell markers
    lines = [
        ln
        for ln in text.splitlines()
        if not ln.startswith("# --- notebook cell")
    ]
    code = "\n".join(lines)
    g = {"__name__": "__main__", "__file__": str(BODY)}
    exec(compile(code, str(BODY), "exec"), g)


if __name__ == "__main__":
    main()
