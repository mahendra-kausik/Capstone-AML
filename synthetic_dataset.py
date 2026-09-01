"""
Synthetic drift dataset generator — validates the SHAP + Kendall-tau diagnostic against
known ground truth, mirroring the Weber-graph `snapshots.pt` contract produced by
Notebooks/01_Elliptic_Preprocessing.ipynb (see CLAUDE.md invariants and data/processed/meta.json).

Two datasets, identical except for one planted change:
  --alpha 0.0  -> NULL   dataset: illicit signal never moves. Diagnostic should stay quiet
                          (tau >= 0.70 at every window transition).
  --alpha 1.0  -> DRIFT  dataset: illicit signal switches from local features (feat_0..93) to
                          aggregated features (feat_94..164) at SHUTDOWN_STEP. Diagnostic should
                          fire (tau << 0.70 at W3->W4), matching the real T43 result.

Per-snapshot node/labelled/edge/illicit counts are reused verbatim from
data/processed/snapshot_stats.csv, so graph size and class imbalance match Elliptic exactly.
Only feature values and edges are synthetic.

Usage:
    python synthetic_dataset.py --selfcheck
    python synthetic_dataset.py --alpha 0.0 --out ../SynthNull
    python synthetic_dataset.py --alpha 1.0 --out ../SynthDrift
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

HERE = os.path.dirname(os.path.abspath(__file__))
STATS_CSV = os.path.join(HERE, "data", "processed", "snapshot_stats.csv")

NUM_FEATURES = 165
LOCAL_MAX = 93          # feat_0..93 local, feat_94..164 aggregated (matches nb04 LOCAL_MAX)
SHUTDOWN_STEP = 43       # matches nb04 SHUTDOWN_STEP; falls inside W4 (T31-49)
N_SIGNAL = 12
MU = 1.2
NOISE_STD = 1.0
WOBBLE_STD = 0.05        # per-snapshot shared feature wobble, keeps NULL non-trivial
EDGE_HOMOPHILY_P = 0.6   # probability an illicit node's edge partner is also illicit


def _sample_signal_dims(rng, low, high, k):
    return rng.choice(np.arange(low, high + 1), size=k, replace=False)


def _make_edges(rng, n_nodes, n_edges_target, illicit_idx_set):
    """Undirected edge set with illicit-illicit homophily, deduplicated, no self-loops."""
    illicit_arr = np.array(sorted(illicit_idx_set), dtype=np.int64)
    have_illicit = len(illicit_arr) >= 2
    pairs = set()
    target = int(n_edges_target)
    attempts = 0
    max_attempts = 60
    while len(pairs) < target and attempts < max_attempts:
        remaining = target - len(pairs)
        batch = max(remaining * 3, 64)
        homophily_mask = rng.random(batch) < EDGE_HOMOPHILY_P if have_illicit else np.zeros(batch, dtype=bool)
        u = np.empty(batch, dtype=np.int64)
        v = np.empty(batch, dtype=np.int64)
        n_h = homophily_mask.sum()
        if n_h:
            u[homophily_mask] = rng.choice(illicit_arr, size=n_h)
            v[homophily_mask] = rng.choice(illicit_arr, size=n_h)
        n_r = batch - n_h
        u[~homophily_mask] = rng.integers(0, n_nodes, size=n_r)
        v[~homophily_mask] = rng.integers(0, n_nodes, size=n_r)
        keep = u != v
        lo = np.minimum(u[keep], v[keep])
        hi = np.maximum(u[keep], v[keep])
        for a, b in zip(lo.tolist(), hi.tolist()):
            pairs.add((a, b))
            if len(pairs) >= target:
                break
        attempts += 1
    pairs = list(pairs)[:target]
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.array(pairs, dtype=np.int64).T  # (2, target)
    undirected = np.concatenate([arr, arr[::-1]], axis=1)
    return undirected


def build_snapshots(alpha, seed=42, n_signal=N_SIGNAL, mu=MU):
    """Returns (list[Data] unscaled, meta dict). Feature scaling happens after, on the full set."""
    stats = pd.read_csv(STATS_CSV)
    assert len(stats) == 49, f"expected 49 snapshot rows, got {len(stats)}"

    rng = np.random.default_rng(seed)
    pre_dims = _sample_signal_dims(rng, 0, LOCAL_MAX, n_signal)
    post_dims = _sample_signal_dims(rng, LOCAL_MAX + 1, NUM_FEATURES - 1, n_signal)
    assert set(pre_dims.tolist()).isdisjoint(post_dims.tolist())

    snapshots = []
    for row in stats.itertuples():
        t = int(row.time_step)
        n = int(row.nodes)
        n_labelled = int(row.labelled)
        n_illicit = int(row.illicit_count)
        n_unknown = n - n_labelled
        n_licit = n_labelled - n_illicit
        assert n_licit >= 0, f"T{t}: labelled ({n_labelled}) < illicit ({n_illicit})"

        y = np.concatenate([
            np.ones(n_illicit, dtype=np.int64),
            np.zeros(n_licit, dtype=np.int64),
            -np.ones(n_unknown, dtype=np.int64),
        ])
        rng.shuffle(y)

        x = rng.normal(0.0, NOISE_STD, size=(n, NUM_FEATURES)).astype(np.float32)
        wobble = rng.normal(0.0, WOBBLE_STD, size=NUM_FEATURES).astype(np.float32)
        x += wobble

        post_shutdown = t >= SHUTDOWN_STEP
        pre_boost = mu * (1.0 - alpha) if post_shutdown else mu
        post_boost = mu * alpha if post_shutdown else 0.0
        illicit_mask = y == 1
        if pre_boost:
            x[np.ix_(illicit_mask, pre_dims)] += pre_boost
        if post_boost:
            x[np.ix_(illicit_mask, post_dims)] += post_boost

        illicit_idx = set(np.nonzero(illicit_mask)[0].tolist())
        edge_index = _make_edges(rng, n, int(row.edges), illicit_idx)

        split = row.split
        train_mask = np.full(n, split == "train")
        val_mask = np.full(n, split == "val")
        test_mask = np.full(n, split == "test")

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            y=torch.from_numpy(y),
            label_mask=torch.from_numpy(y >= 0),
            train_mask=torch.from_numpy(train_mask),
            val_mask=torch.from_numpy(val_mask),
            test_mask=torch.from_numpy(test_mask),
            time_step=torch.tensor(t, dtype=torch.long),
            tx_ids=torch.arange(n, dtype=torch.long),
            num_nodes=n,
        )
        snapshots.append(data)

    meta = {
        "synthetic": True,
        "drift_alpha": alpha,
        "drift_step": SHUTDOWN_STEP,
        "signal_features_pre": pre_dims.tolist(),
        "signal_features_post": post_dims.tolist(),
        "signal_mu": mu,
        "num_snapshots": 49,
        "num_features": NUM_FEATURES,
        "graph_standard": "Weber (unknown nodes kept in graph, masked from loss)",
        "feature_cols": "feat_0 to feat_164",
        "local_features": f"feat_0 to feat_{LOCAL_MAX} ({LOCAL_MAX + 1} features)",
        "aggregated_features": f"feat_{LOCAL_MAX + 1} to feat_{NUM_FEATURES - 1} ({NUM_FEATURES - LOCAL_MAX - 1} features)",
        "total_nodes": int(stats["nodes"].sum()),
        "total_labelled": int(stats["labelled"].sum()),
        "total_unknown": int((stats["nodes"] - stats["labelled"]).sum()),
        "total_edges": int(stats["edges"].sum()),
        "edges_undirected": True,
        "total_illicit": int(stats["illicit_count"].sum()),
        "train_snapshots": "T1-T34",
        "val_snapshots": "T35-T36",
        "test_snapshots": "T37-T49",
        "label_mapping": {"-1": "unknown (masked)", "0": "licit", "1": "illicit"},
        "scaler": "StandardScaler fitted on labelled T1-T34 only",
        "note": "Synthetic ground-truth drift dataset; see CLAUDE.md / PROGRESS.md Phase 7 for context.",
    }
    return snapshots, meta


def scale_snapshots(snapshots):
    train_rows = [
        s.x[s.label_mask & s.train_mask].numpy()
        for s in snapshots
        if (s.label_mask & s.train_mask).any()
    ]
    scaler = StandardScaler().fit(np.concatenate(train_rows, axis=0))
    for s in snapshots:
        s.x = torch.from_numpy(scaler.transform(s.x.numpy()).astype(np.float32))
    return snapshots, scaler


def save_dataset(snapshots, meta, scaler, out_dir):
    processed_dir = os.path.join(out_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(snapshots, os.path.join(processed_dir, "snapshots.pt"))
    with open(os.path.join(processed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(processed_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved -> {processed_dir}")


def selfcheck():
    print("Running self-check (alpha=0.0 NULL, alpha=1.0 DRIFT)...")
    for alpha in (0.0, 1.0):
        snapshots, meta = build_snapshots(alpha)
        assert len(snapshots) == 49
        for s in snapshots:
            assert s.x.shape[1] == NUM_FEATURES
            assert torch.equal(s.label_mask, s.y >= 0)
            u, v = s.edge_index
            assert (u != v).all(), "self-loop found"
            pair_set = set(zip(u.tolist(), v.tolist()))
            for a, b in list(pair_set)[:200]:
                assert (b, a) in pair_set, f"edge ({a},{b}) not symmetric"
            masks = torch.stack([s.train_mask, s.val_mask, s.test_mask])
            assert (masks.sum(0) == 1).all(), "split masks must partition every node exactly once"

        train_snaps = [s for s in snapshots if int(s.time_step) <= 34]
        val_snaps = [s for s in snapshots if 35 <= int(s.time_step) <= 36]
        test_snaps = [s for s in snapshots if int(s.time_step) >= 37]
        assert len(train_snaps) == 34 and len(val_snaps) == 2 and len(test_snaps) == 13
        assert all(bool(s.train_mask.all()) for s in train_snaps)
        assert all(bool(s.val_mask.all()) for s in val_snaps)
        assert all(bool(s.test_mask.all()) for s in test_snaps)

        total_nodes = sum(int(s.num_nodes) for s in snapshots)
        total_labelled = sum(int(s.label_mask.sum()) for s in snapshots)
        total_illicit = sum(int((s.y == 1).sum()) for s in snapshots)
        unknown_frac = 1 - total_labelled / total_nodes
        illicit_frac = total_illicit / total_labelled
        assert 0.75 <= unknown_frac <= 0.79, f"unknown_frac={unknown_frac:.3f}"
        assert 0.09 <= illicit_frac <= 0.11, f"illicit_frac={illicit_frac:.3f}"

        pre = set(meta["signal_features_pre"])
        post = set(meta["signal_features_post"])
        assert pre.issubset(range(0, LOCAL_MAX + 1))
        assert post.issubset(range(LOCAL_MAX + 1, NUM_FEATURES))
        if alpha == 0.0:
            assert pre == pre and post.isdisjoint(pre)  # post dims never boosted; sets untouched
        else:
            assert pre.isdisjoint(post)

        scaled, scaler = scale_snapshots(snapshots)
        lab_train = np.concatenate([
            s.x[s.label_mask & s.train_mask].numpy() for s in scaled
        ], axis=0)
        assert abs(lab_train.mean()) < 0.05, f"scaled train mean={lab_train.mean():.4f}"
        assert abs(lab_train.std() - 1.0) < 0.05, f"scaled train std={lab_train.std():.4f}"

        print(f"  alpha={alpha}: OK  (unknown={unknown_frac:.3f}, illicit={illicit_frac:.3f})")
    print("Self-check passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, help="0.0 = NULL (no drift), 1.0 = DRIFT (local->aggregated at T43)")
    ap.add_argument("--out", type=str, help="output root; writes <out>/data/processed/{snapshots.pt,scaler.pkl,meta.json}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if args.alpha is None or args.out is None:
        ap.error("--alpha and --out are required unless --selfcheck is passed")

    snapshots, meta = build_snapshots(args.alpha, seed=args.seed)
    snapshots, scaler = scale_snapshots(snapshots)
    save_dataset(snapshots, meta, scaler, args.out)


if __name__ == "__main__":
    main()
