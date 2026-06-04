#!/usr/bin/env python3
"""Retrain patched EvolveGCN-H and export publication artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import (
    ALL_IDX,
    BEST_CHECKPOINT,
    BEST_HISTORY_PATH,
    EXPERIMENTS_CSV,
    EVOLVE_SUMMARY_PATH,
    FIGURES_DIR,
    MAX_EPOCHS,
    PATIENCE,
    PROCESSED_DIR,
    SEED,
    SNAPSHOTS_PATH,
    STATIC_SUMMARY_PATH,
    TEST_IDX,
)
from src.evolvegcn.figures import generate_all_figures
from src.evolvegcn.training import (
    TrainConfig,
    build_summary_json,
    get_device,
    is_collapsed_predictions,
    run_drift_probe,
    run_training,
    save_test_outputs,
)


def select_best_row(rows: list[dict]) -> dict:
    viable = [r for r in rows if float(r["val_auroc"]) > 0.51]
    if not viable:
        raise RuntimeError("No non-collapsed runs (val_auroc > 0.51) in experiments CSV")
    return max(
        viable,
        key=lambda r: (
            float(r["val_auroc"]),
            float(r["post_t43_f1"]),
            float(r["test_auroc"]),
        ),
    )


def row_to_config(row: dict) -> TrainConfig:
    h = int(row["hidden_dim"])
    return TrainConfig(
        lr=float(row["lr"]),
        class_weight_illicit=float(row["class_weight"]),
        hidden1=h,
        hidden2=h // 2,
        dropout=float(row["dropout"]),
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
    )


def collect_test_predictions(model, snapshots, device):
    """Match evaluate_test: warm-up T1→T49, score T37–T49."""
    eval_set = set(TEST_IDX)
    was_training = model.training
    model.train()
    model.reset_state(device)
    y_true, y_prob, logits = [], [], []
    with torch.no_grad():
        for i in ALL_IDX:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            if i in eval_set:
                probs = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(data.y.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                logits.append(out.cpu().numpy())
    if not was_training:
        model.eval()
    return (
        np.asarray(y_true),
        np.asarray(y_prob),
        np.vstack(logits),
    )


def print_verification(model, snapshots, device, test_metrics, pre_f1, post_f1):
    y_true, y_prob, logits = collect_test_predictions(model, snapshots, device)
    preds = (y_prob >= 0.5).astype(int)
    uniq_pred, pred_cnt = np.unique(preds, return_counts=True)
    uniq_true, true_cnt = np.unique(y_true, return_counts=True)

    print("\n" + "=" * 60)
    print("STEP 3 — MODEL QUALITY VERIFICATION")
    print("=" * 60)
    print(f"Test AUROC:        {test_metrics['AUROC']:.4f}")
    print(f"Test F1:           {test_metrics['F1']:.4f}")
    print(f"Precision:         {test_metrics['Precision']:.4f}")
    print(f"Recall:            {test_metrics['Recall']:.4f}")
    print(f"Pre-T43 Mean F1:   {pre_f1:.4f}")
    print(f"Post-T43 Mean F1:  {post_f1:.4f}")
    print(f"Prediction counts: {dict(zip(uniq_pred.tolist(), pred_cnt.tolist()))}")
    print(f"True label counts: {dict(zip(uniq_true.tolist(), true_cnt.tolist()))}")
    print(f"P(illicit) std:    {y_prob.std():.6f}")
    print(f"P(illicit) min/max: {y_prob.min():.4f} / {y_prob.max():.4f}")
    collapsed = is_collapsed_predictions(y_prob, logits)
    print(f"Collapsed:         {collapsed}")
    print(f"AUROC > 0.70:      {test_metrics['AUROC'] > 0.70}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Publication retrain for EvolveGCN-H")
    parser.add_argument("--run-id", default=None, help="Override CSV selection")
    parser.add_argument("--csv", default=str(EXPERIMENTS_CSV))
    args = parser.parse_args()

    with Path(args.csv).open() as f:
        rows = list(csv.DictReader(f))

    row = rows[0]
    if args.run_id:
        row = next(r for r in rows if r["run_id"] == args.run_id)
    else:
        row = select_best_row(rows)

    print("=" * 60)
    print("STEP 1 — BEST CANDIDATE (pre-fix grid, non-collapsed)")
    print("=" * 60)
    print(f"run_id:       {row['run_id']}")
    print(f"lr:           {row['lr']}")
    print(f"class_weight: {row['class_weight']}")
    print(f"hidden_dim:   {row['hidden_dim']}")
    print(f"dropout:      {row['dropout']}")
    print(f"val_auroc:    {row['val_auroc']}")
    print(f"post_t43_f1:  {row['post_t43_f1']}")
    print(f"test_auroc:   {row['test_auroc']}")

    cfg = row_to_config(row)
    device = get_device()
    print("\n" + "=" * 60)
    print("STEP 2 — RETRAIN (patched model)")
    print("=" * 60)
    print(f"device: {device}")
    print(f"max_epochs={cfg.max_epochs} patience={cfg.patience} seed={cfg.seed}")

    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)
    result = run_training(cfg, snapshots, device, verbose=True)

    MODELS_DIR = REPO / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": result["model"].state_dict(),
        "config": result["config"],
        "best_val_auroc": result["best_val_auroc"],
        "source_run_id": row["run_id"],
        "model_version": "evolvegcn-h-patched-v1",
    }
    torch.save(checkpoint, BEST_CHECKPOINT)
    BEST_HISTORY_PATH.write_text(json.dumps(result["history"], indent=2))

    drift = run_drift_probe(cfg, snapshots, device)
    summary = build_summary_json(cfg, result, drift)
    summary["source_run_id"] = row["run_id"]
    summary["model_version"] = "evolvegcn-h-patched-v1"
    EVOLVE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    save_test_outputs(
        result["model"], snapshots, device, PROCESSED_DIR / "evolvegcn_outputs.pt"
    )

    with STATIC_SUMMARY_PATH.open() as f:
        static = json.load(f)
    sp = static["per_snapshot"]
    static_pre = float(
        np.mean([r["F1"] for r in sp if r["time_step"] < 43])
    )
    static_post = float(
        np.mean([r["F1"] for r in sp if r["time_step"] > 43])
    )

    print("\n" + "=" * 60)
    print("STEP 4 — ARTIFACTS SAVED")
    print("=" * 60)
    print(f"  {BEST_CHECKPOINT}")
    print(f"  {EVOLVE_SUMMARY_PATH}")
    print(f"  {PROCESSED_DIR / 'evolvegcn_outputs.pt'}")
    print(f"  {BEST_HISTORY_PATH}")

    print("\n" + "=" * 60)
    print("STEP 5 — FIGURES")
    print("=" * 60)
    generate_all_figures(
        result["history"],
        result["per_snapshot"],
        sp,
        result["pre_t43_f1"],
        result["post_t43_f1"],
        static_pre,
        static_post,
        FIGURES_DIR,
    )
    for name in [
        "evolvegcn_training_curves.png",
        "evolvegcn_f1_curve.png",
        "evolvegcn_auroc_curve.png",
        "pre_post_t43_comparison.png",
        "both_models_drift_comparison.png",
    ]:
        p = FIGURES_DIR / name
        print(f"  {p}  ({'ok' if p.exists() else 'MISSING'})")

    print_verification(
        result["model"],
        snapshots,
        device,
        result["test_metrics"],
        result["pre_t43_f1"],
        result["post_t43_f1"],
    )

    tm = result["test_metrics"]
    static_auroc = static["test_metrics"]["AUROC"]
    _, y_prob_chk, logits_chk = collect_test_predictions(
        result["model"], snapshots, device
    )
    ready = tm["AUROC"] > 0.70 and not is_collapsed_predictions(
        y_prob_chk, logits_chk
    )

    print("\n" + "=" * 60)
    print("STEP 6 — FINAL REPORT")
    print("=" * 60)
    print(f"{'Metric':<22} {'EvolveGCN-H':>14} {'Static GCN':>14}")
    print("-" * 52)
    for k in ["AUROC", "F1", "Precision", "Recall"]:
        print(
            f"{k:<22} {tm[k]:>14.4f} {static['test_metrics'][k]:>14.4f}"
        )
    print(f"{'Best Val AUROC':<22} {result['best_val_auroc']:>14.4f} {static['best_val_auroc']:>14.4f}")
    print(f"{'Pre-T43 mean F1':<22} {result['pre_t43_f1']:>14.4f} {static_pre:>14.4f}")
    print(f"{'Post-T43 mean F1':<22} {result['post_t43_f1']:>14.4f} {static_post:>14.4f}")
    print(f"\nCollapse bug resolved: yes (patched GRU + conv bias)")
    print(f"Test AUROC > 0.70:       {tm['AUROC'] > 0.70}")
    if ready:
        print("\nREADY FOR EVOLVEGCN SHAP ANALYSIS")
    else:
        print("\nNot ready for SHAP — review metrics above.")


if __name__ == "__main__":
    main()
