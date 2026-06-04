"""Audit EthereumHeist (UpbitHack) preprocessing artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import (
    ALL_IDX,
    IN_CHANNELS,
    META_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    SCALER_PATH,
    SNAPSHOTS_PATH,
    STATS_PATH,
    TEST_IDX,
    TRAIN_IDX,
    VAL_IDX,
)


def _check_snapshots(snapshots: list) -> dict[str, Any]:
    issues: list[str] = []
    rows: list[dict] = []

    if len(snapshots) != 49:
        issues.append(f"expected 49 snapshots, got {len(snapshots)}")

    prev_t = 0
    for i, s in enumerate(snapshots):
        t = int(s.time_step.item())
        if t != i + 1:
            issues.append(f"index {i}: time_step={t}, expected {i+1}")
        if prev_t and t < prev_t:
            issues.append(f"non-monotonic time_step at index {i}")
        prev_t = t

        n = s.num_nodes
        feat_dim = s.x.shape[1] if s.num_nodes else IN_CHANNELS
        if feat_dim != IN_CHANNELS:
            issues.append(f"T{t}: feature dim {feat_dim} != {IN_CHANNELS}")

        labels_ok = hasattr(s, "y") and s.y is not None
        split_flags = (
            hasattr(s, "train_mask")
            and hasattr(s, "val_mask")
            and hasattr(s, "test_mask")
        )
        illicit = int(s.y.sum().item()) if labels_ok and n else 0

        rows.append(
            {
                "time_step": t,
                "nodes": n,
                "edges": int(s.edge_index.shape[1]),
                "illicit": illicit,
                "illicit_pct": round(100 * illicit / max(n, 1), 2),
                "feat_dim": feat_dim,
                "has_labels": labels_ok,
                "has_masks": split_flags,
            }
        )

    # Split mask semantics (snapshot-level)
    for i in TRAIN_IDX:
        s = snapshots[i]
        if s.num_nodes and not bool(s.train_mask.all()):
            issues.append(f"T{i+1}: expected all-train mask")
    for i in VAL_IDX:
        s = snapshots[i]
        if s.num_nodes and not bool(s.val_mask.all()):
            issues.append(f"T{i+1}: expected all-val mask")
    for i in TEST_IDX:
        s = snapshots[i]
        if s.num_nodes and not bool(s.test_mask.all()):
            issues.append(f"T{i+1}: expected all-test mask")

    total_nodes = sum(r["nodes"] for r in rows)
    total_edges = sum(r["edges"] for r in rows)
    total_illicit = sum(r["illicit"] for r in rows)

    return {
        "n_snapshots": len(snapshots),
        "temporal_ok": not any("time_step" in x or "monotonic" in x for x in issues),
        "feature_dim_ok": all(r["feat_dim"] == IN_CHANNELS for r in rows),
        "labels_ok": all(r["has_labels"] for r in rows),
        "split_masks_ok": all(r["has_masks"] for r in rows),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_illicit": total_illicit,
        "overall_illicit_pct": round(100 * total_illicit / max(total_nodes, 1), 3),
        "per_snapshot": rows,
        "issues": issues,
    }


def audit_upbit_dataset() -> dict[str, Any]:
    """Run full audit; returns JSON-serializable report."""
    report: dict[str, Any] = {
        "processed_dir": str(PROCESSED_DIR),
        "raw_dir": str(RAW_DIR),
        "artifacts": {},
        "checks": {},
        "reference_colab_summary": {
            "num_snapshots": 49,
            "num_features": 165,
            "total_nodes": 761448,
            "total_edges": 2318459,
            "total_illicit": 71033,
            "overall_illicit_pct": 9.329,
            "train_snapshots": "T1–T34",
            "val_snapshots": "T35–T36",
            "test_snapshots": "T37–T49",
            "class_imbalance_train": "1:11.4",
            "source": "UpbitHack — EthereumHeist (IEEE TIFS 2023)",
        },
    }

    for name, path in [
        ("snapshots.pt", SNAPSHOTS_PATH),
        ("meta.json", META_PATH),
        ("scaler.pkl", SCALER_PATH),
        ("snapshot_stats.csv", STATS_PATH),
    ]:
        report["artifacts"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }

    raw_files = ["all-tx.csv", "all-address.csv", "accounts-hacker.csv"]
    report["raw_csvs"] = {
        f: (RAW_DIR / f).exists() for f in raw_files
    }

    if not SNAPSHOTS_PATH.exists():
        report["status"] = "MISSING_SNAPSHOTS"
        report["message"] = (
            "processed_upbit/snapshots.pt not found. Copy from Colab Drive "
            "(data/processed_upbit/) or run: python scripts/preprocess_upbit.py "
            "after placing CSVs in data/raw/upbit/"
        )
        return report

    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)
    snap_audit = _check_snapshots(snapshots)
    report["checks"]["snapshots"] = snap_audit

    if META_PATH.exists():
        report["meta"] = json.loads(META_PATH.read_text())

    report["status"] = "PASS" if not snap_audit["issues"] else "WARN"
    if snap_audit["issues"]:
        report["status"] = "FAIL"
    return report


def format_summary_table(report: dict[str, Any]) -> str:
    """Markdown dataset summary table for Step 1."""
    lines = [
        "# UpbitHack (EthereumHeist) — Dataset Audit",
        "",
        f"**Status:** `{report.get('status', 'UNKNOWN')}`",
        f"**Processed dir:** `{report.get('processed_dir', '')}`",
        "",
        "## Artifact inventory",
        "",
        "| File | Present | Size (bytes) |",
        "|------|---------|--------------|",
    ]
    for name, info in report.get("artifacts", {}).items():
        exists = "✓" if info.get("exists") else "✗"
        size = info.get("size_bytes") if info.get("exists") else "—"
        lines.append(f"| {name} | {exists} | {size} |")

    lines.extend(
        [
            "",
            "## Raw CSVs (for re-preprocessing)",
            "",
            "| File | Present |",
            "|------|---------|",
        ]
    )
    for f, ok in report.get("raw_csvs", {}).items():
        lines.append(f"| {f} | {'✓' if ok else '✗'} |")

    chk = report.get("checks", {}).get("snapshots")
    if chk:
        lines.extend(
            [
                "",
                "## Loaded snapshots verification",
                "",
                "| Check | Result |",
                "|-------|--------|",
                f"| Snapshots | {chk['n_snapshots']} |",
                f"| Feature dim (165) | {'✓' if chk['feature_dim_ok'] else '✗'} |",
                f"| Labels (y) | {'✓' if chk['labels_ok'] else '✗'} |",
                f"| Train/val/test masks | {'✓' if chk['split_masks_ok'] else '✗'} |",
                f"| Temporal order | {'✓' if chk['temporal_ok'] else '✗'} |",
                f"| Total nodes | {chk['total_nodes']:,} |",
                f"| Total edges | {chk['total_edges']:,} |",
                f"| Total illicit | {chk['total_illicit']:,} |",
                f"| Illicit % | {chk['overall_illicit_pct']}% |",
                "",
                "### Train / val / test split",
                "",
                "| Split | Snapshot indices | Time steps |",
                "|-------|------------------|------------|",
                f"| Train | 0–33 | T1–T34 |",
                f"| Val | 34–35 | T35–T36 |",
                f"| Test | 36–48 | T37–T49 |",
            ]
        )
        if chk.get("issues"):
            lines.append("\n**Issues:**\n")
            for issue in chk["issues"]:
                lines.append(f"- {issue}")

    ref = report.get("reference_colab_summary", {})
    if ref:
        lines.extend(
            [
                "",
                "## Reference (Colab preprocessing run)",
                "",
                "| Field | Colab value | Local value |",
                "|-------|-------------|-------------|",
            ]
        )
        local = chk or {}
        pairs = [
            ("Snapshots", ref.get("num_snapshots"), local.get("n_snapshots")),
            ("Features", ref.get("num_features"), IN_CHANNELS),
            ("Total nodes", ref.get("total_nodes"), local.get("total_nodes")),
            ("Total edges", ref.get("total_edges"), local.get("total_edges")),
            ("Total illicit", ref.get("total_illicit"), local.get("total_illicit")),
            ("Illicit %", ref.get("overall_illicit_pct"), local.get("overall_illicit_pct")),
        ]
        for label, colab, loc in pairs:
            loc_s = f"{loc:,}" if isinstance(loc, int) else (loc if loc is not None else "—")
            col_s = f"{colab:,}" if isinstance(colab, int) else colab
            lines.append(f"| {label} | {col_s} | {loc_s} |")

    if report.get("message"):
        lines.extend(["", f"**Note:** {report['message']}"])

    return "\n".join(lines) + "\n"
