"""Training, evaluation, and experiment utilities for EvolveGCN-H."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import (
    ALL_IDX,
    DRIFT_TEST_IDX,
    DRIFT_TRAIN_IDX,
    GRAD_CLIP,
    MAX_EPOCHS,
    PATIENCE,
    SHUTDOWN_STEP,
    TEST_IDX,
    TRAIN_IDX,
    VAL_IDX,
    WEIGHT_DECAY,
)
from .model import EvolveGCNH


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _forward_probs(
    model: EvolveGCNH,
    snapshots: list,
    idx_list: list[int],
    device: torch.device,
    *,
    initial_state: tuple | None = None,
    reset_first: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass for metric collection.

    EvolveGCN-H is trained with dropout active. In eval() mode, dropout is off and
    the network often emits identical logits for every node (AUROC=0.5, Recall=1).
    We keep train() mode for the forward pass but disable gradients.
    """
    was_training = model.training
    model.train()
    if reset_first:
        model.reset_state(device)
    if initial_state is not None:
        model.set_state(initial_state)

    y_true, y_prob, logits = [], [], []
    with torch.no_grad():
        for i in idx_list:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]
            y_true.extend(data.y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            logits.append(out.cpu().numpy())

    if not was_training:
        model.eval()
    return np.asarray(y_true), np.asarray(y_prob), np.vstack(logits)


def _val_forward_probs(
    model: EvolveGCNH,
    snapshots: list,
    val_idx: list[int],
    end_state: tuple,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate on val snapshots continuing from post-train W_state."""
    return _forward_probs(
        model, snapshots, val_idx, device, initial_state=end_state, reset_first=False
    )


def is_collapsed_predictions(y_prob: np.ndarray, logits: np.ndarray) -> bool:
    """
    True when the model assigns (near) identical scores to every node.
    sklearn AUROC is 0.5 for constant scores; argmax may still look 'confident'.
    """
    if y_prob.size == 0:
        return True
    prob_std = float(np.std(y_prob))
    logit_std = float(np.std(logits[:, 1] - logits[:, 0]))
    return prob_std < 1e-4 and logit_std < 1e-4


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    class_weight_illicit: float = 9.0
    hidden1: int = 64
    hidden2: int = 32
    dropout: float = 0.5
    grad_clip: float = GRAD_CLIP
    seed: int = 42

    @property
    def architecture_str(self) -> str:
        return f"165 → {self.hidden1} → {self.hidden2} → 2  (GRU-evolved)"


def build_model(cfg: TrainConfig, device: torch.device) -> EvolveGCNH:
    model = EvolveGCNH(
        in_channels=165,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
    ).to(device)
    model.reset_weights()
    return model


def train_epoch(
    model: EvolveGCNH,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    snapshots: list,
    idx_list: list[int],
    device: torch.device,
    grad_clip: float = GRAD_CLIP,
) -> tuple[float, tuple]:
    model.train()
    model.reset_state(device)
    total_loss = 0.0
    for i in idx_list:
        data = snapshots[i].to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(idx_list), model.get_state()


def evaluate_val(
    model: EvolveGCNH,
    snapshots: list,
    val_idx: list[int],
    end_state: tuple,
    device: torch.device,
) -> dict[str, float]:
    y_true, y_prob, logits = _val_forward_probs(model, snapshots, val_idx, end_state, device)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _metrics_from_arrays(y_true, y_pred, y_prob)
    metrics["collapsed"] = is_collapsed_predictions(y_prob, logits)
    metrics["prob_std"] = float(np.std(y_prob))
    return metrics


def evaluate_val_loss(
    model: EvolveGCNH,
    snapshots: list,
    val_idx: list[int],
    end_state: tuple,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Mean CrossEntropyLoss on val snapshots (W_state warm-started from train epoch)."""
    was_training = model.training
    model.train()
    model.set_state(end_state)
    total_loss = 0.0
    with torch.no_grad():
        for i in val_idx:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            total_loss += criterion(out, data.y).item()
    if not was_training:
        model.eval()
    return total_loss / len(val_idx)


def evaluate_test(
    model: EvolveGCNH,
    snapshots: list,
    all_idx: list[int],
    eval_idx: list[int],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.train()
    model.reset_state(device)
    eval_set = set(eval_idx)
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for i in all_idx:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            if i in eval_set:
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
    if not was_training:
        model.eval()
    return _metrics_from_arrays(y_true, y_pred, y_prob)


def evaluate_per_snapshot(
    model: EvolveGCNH,
    snapshots: list,
    all_idx: list[int],
    eval_idx: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    eval_set = set(eval_idx)
    results = []
    was_training = model.training
    model.train()
    model.reset_state(device)
    with torch.no_grad():
        for i in all_idx:
            data = snapshots[i].to(device)
            out, _ = model(data.x, data.edge_index)
            if i in eval_set:
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
                m = _metrics_from_arrays(
                    data.y.cpu().numpy(),
                    preds.cpu().numpy(),
                    probs.cpu().numpy(),
                )
                results.append({"time_step": int(snapshots[i].time_step.item()), **m})
    if not was_training:
        model.eval()
    return results


def pre_post_t43_f1(per_snapshot: list[dict]) -> tuple[float, float]:
    pre = [r["F1"] for r in per_snapshot if r["time_step"] < SHUTDOWN_STEP]
    post = [r["F1"] for r in per_snapshot if r["time_step"] > SHUTDOWN_STEP]
    return float(np.mean(pre)), float(np.mean(post))


def run_training(
    cfg: TrainConfig,
    snapshots: list,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    set_seed(cfg.seed)

    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.class_weight_illicit], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_auroc = -1.0
    best_state_dict = None
    best_end_state = None
    patience_counter = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_auroc_history: list[float] = []
    val_f1_history: list[float] = []

    for epoch in range(cfg.max_epochs):
        loss, end_state = train_epoch(
            model,
            optimizer,
            criterion,
            snapshots,
            TRAIN_IDX,
            device,
            grad_clip=cfg.grad_clip,
        )
        val_met = evaluate_val(model, snapshots, VAL_IDX, end_state, device)
        val_loss = evaluate_val_loss(
            model, snapshots, VAL_IDX, end_state, criterion, device
        )
        train_loss_history.append(loss)
        val_loss_history.append(val_loss)
        val_auroc_history.append(val_met["AUROC"])
        val_f1_history.append(val_met["F1"])

        val_auroc = val_met["AUROC"]
        collapsed = val_met.get("collapsed", False)
        improved = (
            not np.isnan(val_auroc)
            and not collapsed
            and val_auroc > 0.51  # strictly better than random / constant scores
            and val_auroc > best_val_auroc + 1e-4
        )
        if improved:
            best_val_auroc = val_auroc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_end_state = end_state
            patience_counter = 0
            status = "saved"
        else:
            patience_counter += 1
            if collapsed:
                status = f"collapsed (P_std={val_met.get('prob_std', 0):.2e})"
            else:
                status = f"patience {patience_counter}/{cfg.patience}"

        if verbose and (epoch % 10 == 0 or patience_counter >= cfg.patience):
            print(
                f"  epoch {epoch:>4}  train_loss={loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_f1={val_met['F1']:.4f}  val_auroc={val_auroc:.4f}  "
                f"P_std={val_met.get('prob_std', 0):.4f}  {status}"
            )

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  early stop @ epoch {epoch}")
            break

    if best_state_dict is None:
        raise RuntimeError(
            "No valid checkpoint saved — all epochs had collapsed predictions or AUROC≤0.51. "
            "Lower LR (try 1e-4), reduce CLASS_WEIGHT, or train longer before early stopping."
        )

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_test(model, snapshots, ALL_IDX, TEST_IDX, device)
    per_snapshot = evaluate_per_snapshot(model, snapshots, ALL_IDX, TEST_IDX, device)
    pre_f1, post_f1 = pre_post_t43_f1(per_snapshot)

    return {
        "model": model,
        "best_val_auroc": float(best_val_auroc),
        "test_metrics": test_metrics,
        "per_snapshot": per_snapshot,
        "pre_t43_f1": pre_f1,
        "post_t43_f1": post_f1,
        "history": {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "val_auroc": val_auroc_history,
            "val_f1": val_f1_history,
        },
        "best_end_state": best_end_state,
        "config": asdict(cfg),
    }


def run_drift_probe(
    cfg: TrainConfig,
    snapshots: list,
    device: torch.device,
) -> dict[str, float]:
    """Train on T1–T16, evaluate with chronological warm-up T1→T49 at probe snapshots."""
    global cfg_grad_clip
    cfg_grad_clip = cfg.grad_clip
    set_seed(cfg.seed + 1)

    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.class_weight_illicit], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.max_epochs):
        train_epoch(
            model,
            optimizer,
            criterion,
            snapshots,
            DRIFT_TRAIN_IDX,
            device,
            grad_clip=cfg.grad_clip,
        )

    probe_results = evaluate_per_snapshot(
        model, snapshots, ALL_IDX, DRIFT_TEST_IDX, device
    )
    pre, post = pre_post_t43_f1(probe_results)
    return {
        "mean_f1_pre_t43": pre,
        "mean_f1_post_t43": post,
        "f1_drop": pre - post,
        "per_snapshot": probe_results,
    }


def build_summary_json(
    cfg: TrainConfig,
    run_result: dict[str, Any],
    drift_probe: dict[str, float],
) -> dict[str, Any]:
    tm = run_result["test_metrics"]
    per = run_result["per_snapshot"]
    return {
        "model": "EvolveGCN-H",
        "architecture": cfg.architecture_str,
        "train_snapshots": "T1-T34",
        "val_snapshots": "T35-T36",
        "test_snapshots": "T37-T49",
        "best_val_auroc": round(run_result["best_val_auroc"], 4),
        "hyperparameters": asdict(cfg),
        "test_metrics": {k: round(v, 4) for k, v in tm.items()},
        "per_snapshot": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in per
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


def save_test_outputs(model: EvolveGCNH, snapshots: list, device: torch.device, path) -> None:
    model.eval()
    all_outputs = {}
    with torch.no_grad():
        for i in TEST_IDX:
            data = snapshots[i].to(device)
            out, emb = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_outputs[i] = {
                "time_step": int(snapshots[i].time_step.item()),
                "preds": out.argmax(dim=1).cpu(),
                "probs": probs.cpu(),
                "embeddings": emb.cpu(),
                "y_true": data.y.cpu(),
            }
    torch.save(all_outputs, path)
