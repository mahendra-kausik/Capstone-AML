"""Training and evaluation for Static GCN."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.evolvegcn.config import SHUTDOWN_STEP
from src.evolvegcn.training import get_device, set_seed
from src.static_gcn.model import StaticGCN
from src.upbit.config import (
    ALL_IDX,
    DRIFT_TEST_IDX,
    DRIFT_TRAIN_IDX,
    IN_CHANNELS,
    TEST_IDX,
    TRAIN_IDX,
    VAL_IDX,
)


def _metrics_from_arrays(y_true, y_pred, y_prob) -> dict[str, float]:
    if len(set(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
    return {
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUROC": auroc,
        "Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


@dataclass
class StaticTrainConfig:
    lr: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 40
    class_weight_illicit: float = 9.0
    hidden1: int = 64
    hidden2: int = 32
    dropout: float = 0.5
    seed: int = 42

    @property
    def architecture_str(self) -> str:
        return f"{IN_CHANNELS} → {self.hidden1} → {self.hidden2} → 2"


def train_epoch(model, optimizer, criterion, snapshots, idx_list, device):
    model.train()
    total_loss = 0.0
    for i in idx_list:
        data = snapshots[i].to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(idx_list)


def evaluate(model, snapshots, idx_list, device) -> dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for i in idx_list:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return _metrics_from_arrays(y_true, y_pred, y_prob)


def evaluate_per_snapshot(model, snapshots, idx_list, device) -> list[dict]:
    results = []
    for i in idx_list:
        m = evaluate(model, snapshots, [i], device)
        results.append({"time_step": int(snapshots[i].time_step.item()), **m})
    return results


def pre_post_t43_f1(per_snapshot: list[dict]) -> tuple[float, float]:
    pre = [r["F1"] for r in per_snapshot if r["time_step"] < SHUTDOWN_STEP]
    post = [r["F1"] for r in per_snapshot if r["time_step"] > SHUTDOWN_STEP]
    return float(np.mean(pre)), float(np.mean(post))


def run_static_training(
    cfg: StaticTrainConfig,
    snapshots: list,
    device: torch.device,
    checkpoint_path: Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    model = StaticGCN(
        in_channels=IN_CHANNELS,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.class_weight_illicit], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_auroc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        loss = train_epoch(model, optimizer, criterion, snapshots, TRAIN_IDX, device)
        val_met = evaluate(model, snapshots, VAL_IDX, device)
        val_auroc = val_met["AUROC"]

        improved = not np.isnan(val_auroc) and val_auroc > best_val_auroc + 1e-4
        if improved:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "saved"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{cfg.patience}"

        if verbose and (epoch % 10 == 0 or patience_counter >= cfg.patience):
            print(
                f"  epoch {epoch:>4}  loss={loss:.4f}  val_f1={val_met['F1']:.4f}  "
                f"val_auroc={val_auroc:.4f}  {status}"
            )
        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  early stop @ epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Static GCN: no checkpoint saved (val AUROC never improved)")

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)

    test_metrics = evaluate(model, snapshots, TEST_IDX, device)
    per_snapshot = evaluate_per_snapshot(model, snapshots, TEST_IDX, device)
    pre_f1, post_f1 = pre_post_t43_f1(per_snapshot)

    return {
        "model": model,
        "best_val_auroc": float(best_val_auroc),
        "test_metrics": test_metrics,
        "per_snapshot": per_snapshot,
        "pre_t43_f1": pre_f1,
        "post_t43_f1": post_f1,
        "config": asdict(cfg),
    }


def run_static_drift_probe(cfg: StaticTrainConfig, snapshots: list, device: torch.device) -> dict:
    set_seed(cfg.seed + 1)
    model = StaticGCN(
        in_channels=IN_CHANNELS,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.class_weight_illicit], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    for _ in range(cfg.max_epochs):
        train_epoch(model, optimizer, criterion, snapshots, DRIFT_TRAIN_IDX, device)
    probe = evaluate_per_snapshot(model, snapshots, DRIFT_TEST_IDX, device)
    pre, post = pre_post_t43_f1(probe)
    return {"mean_f1_pre_t43": pre, "mean_f1_post_t43": post, "f1_drop": pre - post}


def build_static_summary(
    cfg: StaticTrainConfig,
    run_result: dict[str, Any],
    drift_probe: dict[str, float],
    *,
    dataset: str = "UpbitHack",
) -> dict[str, Any]:
    tm = run_result["test_metrics"]
    return {
        "model": "Static GCN",
        "dataset": dataset,
        "architecture": cfg.architecture_str,
        "train_snapshots": "T1-T34",
        "val_snapshots": "T35-T36",
        "test_snapshots": "T37-T49",
        "best_val_auroc": round(run_result["best_val_auroc"], 4),
        "hyperparameters": asdict(cfg),
        "test_metrics": {k: round(v, 4) for k, v in tm.items()},
        "per_snapshot": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in run_result["per_snapshot"]
        ],
        "pre_t43_mean_f1": round(run_result["pre_t43_f1"], 4),
        "post_t43_mean_f1": round(run_result["post_t43_f1"], 4),
        "f1_drop": round(run_result["pre_t43_f1"] - run_result["post_t43_f1"], 4),
        "drift_probe": {
            "train": "T1-T16",
            "test": "T33-T49",
            "mean_f1_pre_t43": round(drift_probe["mean_f1_pre_t43"], 4),
            "mean_f1_post_t43": round(drift_probe["mean_f1_post_t43"], 4),
            "f1_drop": round(drift_probe["f1_drop"], 4),
        },
    }


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
