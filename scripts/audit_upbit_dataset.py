#!/usr/bin/env python3
"""Step 1: Audit UpbitHack preprocessing artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.upbit.audit import audit_upbit_dataset, format_summary_table
from src.upbit.config import AUDIT_PATH, AUDIT_TABLE_PATH, RESULTS_DIR


def main():
    report = audit_upbit_dataset()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(report, indent=2))
    AUDIT_TABLE_PATH.write_text(format_summary_table(report))

    print(format_summary_table(report))
    print(f"\nSaved {AUDIT_PATH}")
    print(f"Saved {AUDIT_TABLE_PATH}")

    if report.get("status") == "MISSING_SNAPSHOTS":
        sys.exit(2)
    if report.get("status") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
