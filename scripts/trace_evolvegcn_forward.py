#!/usr/bin/env python3
"""Trace where node-wise variance collapses in EvolveGCN-H forward pass."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import SNAPSHOTS_PATH, TRAIN_IDX, VAL_IDX
from src.evolvegcn.model import EvolveGCNH, EvolveGCNLayer
from src.evolvegcn.training import build_model, train_epoch, TrainConfig, set_seed
import torch.nn as nn


def stats(name, t, indent=0):
    if t is None:
        print("  " * indent + f"{name}: None")
        return
    if t.dim() == 0:
        print("  " * indent + f"{name}: scalar {t.item():.6f}")
        return
    print(
        "  " * indent
        + f"{name}: shape={tuple(t.shape)} mean={t.mean().item():.6f} std={t.std().item():.6f} "
        + f"min={t.min().item():.6f} max={t.max().item():.6f}"
    )
    if t.shape[0] > 1:
        row_std = t.std(dim=0).mean().item() if t.dim() > 1 else t.std().item()
        print("  " * indent + f"       per-node/std(dim0)_mean={row_std:.6f}")


def trace_layer(layer: EvolveGCNLayer, x, edge_index, name: str, device):
    print(f"\n{'='*70}\n{name}  in_features={layer.in_channels} out={layer.out_channels}\n{'='*70}")
    stats("input x", x, 0)

    if layer.W_state is None:
        layer.reset_state(device)
    stats("W_state (before)", layer.W_state, 0)
    if layer.W_state is not None and layer.W_state.shape[0] > 1:
        stats("W_state row0 vs row1 diff", (layer.W_state[0] - layer.W_state[1]).abs(), 0)

    w_in = layer.W_state.detach()
    stats("w_in", w_in, 0)

    z = x.mean(dim=0)
    stats("z (graph mean)", z, 0)
    z_rep = z.unsqueeze(0).repeat(layer.out_channels, 1)
    stats("z_rep (repeated rows)", z_rep, 0)
    if z_rep.shape[0] > 1:
        stats("z_rep row equality", (z_rep[0] - z_rep[1]).abs(), 0)

    w_evolved = layer.gru(z_rep, w_in)
    stats("w_evolved (after GRU)", w_evolved, 0)
    if w_evolved.shape[0] > 1:
        stats("w_evolved row0-row1", (w_evolved[0] - w_evolved[1]).abs(), 0)
        row_stds = w_evolved.std(dim=1)
        stats("w_evolved per-row std (across cols)", row_stds, 0)

    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edge_index_sl
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
    agg.scatter_add_(
        0,
        col.unsqueeze(1).expand(-1, x.size(1)),
        norm.unsqueeze(1) * x[row],
    )
    stats("agg (GCN aggregate)", agg, 0)
    if agg.shape[0] > 1:
        stats("agg per-node L2 norm", agg.norm(dim=1), 0)

    out = F.linear(agg, w_evolved)
    stats("layer output F.linear(agg, W)", out, 0)
    if out.shape[0] > 1:
        stats("output unique rows check std(dim=0)", out.std(dim=0), 0)

    layer.W_state = w_evolved.detach()
    return out


def trace_model(model: EvolveGCNH, x, edge_index, device, label: str):
    print(f"\n\n########## {label} ##########")
    model.reset_state(device)
    h1 = trace_layer(model.layer1, x, edge_index, "Layer1", device)
    h1r = F.relu(h1)
    stats("after ReLU-1", h1r, 0)
    h1d = F.dropout(h1r, p=model.dropout, training=model.training)
    stats(f"after Dropout-1 (training={model.training})", h1d, 0)

    h2 = trace_layer(model.layer2, h1d, edge_index, "Layer2", device)
    h2r = F.relu(h2)
    stats("after ReLU-2", h2r, 0)
    logits = model.classifier(h2r)
    stats("logits", logits, 0)
    probs = torch.softmax(logits, dim=1)[:, 1]
    stats("P(illicit)", probs, 0)
    return logits, probs


def main():
    device = torch.device("cpu")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)
    set_seed(42)

    # --- Untrained ---
    m = build_model(TrainConfig(), device)
    d = snapshots[VAL_IDX[0]]
    trace_model(m, d.x, d.edge_index, device, "UNTRAINED eval()")
    m.train()
    trace_model(m, d.x, d.edge_index, device, "UNTRAINED train()")

    # --- After 1 train epoch (collapsed case) ---
    m2 = build_model(TrainConfig(lr=1e-4, class_weight_illicit=9.0), device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    loss, es = train_epoch(m2, opt, crit, snapshots, TRAIN_IDX, device)
    print(f"\n\nTrain epoch loss={loss:.4f}")

    m2.eval()
    m2.set_state(es)
    d = snapshots[VAL_IDX[0]]
    trace_model(m2, d.x, d.edge_index, device, "AFTER TRAIN eval() on T35")

    m2.train()
    m2.set_state(es)
    trace_model(m2, d.x, d.edge_index, device, "AFTER TRAIN train() on T35")

    # --- Isolate: same agg different W ---
    print("\n\n########## ISOLATION: linear sensitivity ##########")
    m2.eval()
    m2.reset_state(device)
    with torch.no_grad():
        h1 = m2.layer1(d.x, d.edge_index)
    stats("h1 from layer1 only", h1, 0)

    # Manual identical W test
    W_bad = torch.zeros(64, 165)
    W_bad[:] = m2.layer1.W_state[0]  # all rows same
    agg_test = torch.randn(100, 165)
    out_bad = F.linear(agg_test, W_bad)
    stats("F.linear with identical W rows", out_bad, 0)

    W_good = torch.randn(64, 165)
    out_good = F.linear(agg_test, W_good)
    stats("F.linear with random W", out_good, 0)

    # GRU: identical z effect
    print("\n\n########## GRU: z identical all rows ##########")
    layer = EvolveGCNLayer(165, 64)
    layer.reset_state(device)
    w_in = layer.W_state
    z = d.x.mean(0).unsqueeze(0).repeat(64, 1)
    w1 = layer.gru(z, w_in)
    z2 = d.x.mean(0).unsqueeze(0).repeat(64, 1)  # same z
    w2 = layer.gru(z2, w1)
    stats("w1-w2 row diff after 2 GRU steps same z", (w1 - w2).abs(), 0)


if __name__ == "__main__":
    main()
