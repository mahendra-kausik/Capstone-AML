#!/usr/bin/env python3
"""Diagnostics for EvolveGCN-H AUROC=0.5 / collapsed predictions."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import SNAPSHOTS_PATH, TRAIN_IDX, VAL_IDX, TEST_IDX
from src.evolvegcn.training import (
    TrainConfig,
    build_model,
    train_epoch,
    evaluate_val,
    evaluate_test,
    set_seed,
    _val_forward_probs,
    is_collapsed_predictions,
)


def split_diagnostics(name, idx_list, snapshots):
    print(f"\n{'='*60}\n{name}")
    y_all = []
    for i in idx_list:
        y = snapshots[i].y.cpu().numpy()
        y_all.extend(y.tolist())
        uniq, cnt = np.unique(y, return_counts=True)
        print(f"  T{snapshots[i].time_step.item():>2}  n={len(y):>5}  {dict(zip(uniq.tolist(), cnt.tolist()))}")
    y_all = np.array(y_all)
    print(f"  POOLED licit={int((y_all==0).sum())} illicit={int((y_all==1).sum())} unique={np.unique(y_all).tolist()}")


def print_forward_stats(label, y_true, y_prob, logits):
    preds = (y_prob >= 0.5).astype(int)
    print(f"\n--- {label} ---")
    print(f"  y_true: {np.unique(y_true, return_counts=True)}")
    print(f"  preds:  {np.unique(preds, return_counts=True)}")
    print(f"  logit[0] mean={logits[:,0].mean():.4f} std={logits[:,0].std():.4f}")
    print(f"  logit[1] mean={logits[:,1].mean():.4f} std={logits[:,1].std():.4f}")
    print(f"  P(illicit) mean={y_prob.mean():.4f} std={y_prob.std():.4f} min={y_prob.min():.4f} max={y_prob.max():.4f}")
    print(f"  collapsed={is_collapsed_predictions(y_prob, logits)}")
    if len(np.unique(y_true)) >= 2:
        from sklearn.metrics import roc_auc_score
        print(f"  AUROC(prob)={roc_auc_score(y_true, y_prob):.4f}")


def main():
    device = torch.device("cpu")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)

    split_diagnostics("TRAIN", TRAIN_IDX, snapshots)
    split_diagnostics("VAL", VAL_IDX, snapshots)

    cfg = TrainConfig(lr=1e-4, class_weight_illicit=9.0, dropout=0.3, seed=42)
    set_seed(cfg.seed)
    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cfg.class_weight_illicit]))

    model.eval()
    model.reset_state(device)
    for tag, idx in [("VAL epoch0-train", VAL_IDX)]:
        yt, yp, lg = _val_forward_probs(model, snapshots, idx, model.get_state(), device)
        print_forward_stats(tag + " (untrained)", yt, yp, lg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for ep in [1, 5, 10, 20]:
        for _ in range(ep if ep == 1 else ep - getattr(main, '_last', 0)):
            train_epoch(model, opt, criterion, snapshots, TRAIN_IDX, device)
        main._last = ep
        loss, es = train_epoch(model, opt, criterion, snapshots, TRAIN_IDX, device)
        vm = evaluate_val(model, snapshots, VAL_IDX, es, device)
        yt, yp, lg = _val_forward_probs(model, snapshots, VAL_IDX, es, device)
        print(f"\n>>> after {ep} train epochs loss={loss:.4f} metrics={vm}")
        print_forward_stats(f"VAL after {ep} epochs", yt, yp, lg)

    tm = evaluate_test(model, snapshots, list(range(49)), TEST_IDX, device)
    print(f"\nTEST pooled: {tm}")


if __name__ == "__main__":
    main._last = 0
    main()
