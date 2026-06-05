#!/usr/bin/env python3
"""Train tabular baseline models on Elliptic with the same temporal split as Notebook 02."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.config import (
    SEED,
    SNAPSHOTS_PATH,
    TEST_IDX,
    TRAIN_IDX,
    VAL_IDX,
)

PAPER_DIR = REPO / "paper"
RESULTS_PATH = REPO / "results" / "baseline_comparison.json"

CLASS_WEIGHT_ILLICIT = 9.0
XGB_PYTHON_CANDIDATES = (
    "/opt/homebrew/bin/python3.11",
    "/usr/local/bin/python3.12",
    "/usr/local/bin/python3.11",
    sys.executable,
)


def metrics_from_arrays(y_true, y_pred, y_prob) -> dict[str, float]:
    if len(set(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
    return {
        "AUROC": round(auroc, 4),
        "F1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }


def collect_split(snapshots, idx_list) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in idx_list:
        data = snapshots[i]
        xs.append(data.x.numpy())
        ys.append(data.y.numpy())
    return np.vstack(xs), np.concatenate(ys)


class MLPClassifier(nn.Module):
    """165 → 64 → 32 → 2 tabular MLP (no graph structure)."""

    def __init__(self, in_dim: int = 165, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train, y_train, X_val, y_val, device, seed: int = SEED
) -> MLPClassifier:
    torch.manual_seed(seed)
    model = MLPClassifier().to(device)
    weight = torch.tensor([1.0, CLASS_WEIGHT_ILLICIT], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    best_auroc = -1.0
    best_state = None
    patience = 20
    counter = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_val_t), dim=1)[:, 1].cpu().numpy()
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()
            val_m = metrics_from_arrays(y_val, preds, probs)
            val_auroc = val_m["AUROC"]

        if not np.isnan(val_auroc) and val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model, X, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(X_t)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    return preds, probs


def train_xgboost(
    X_train, y_train, X_val, y_val, X_test, y_test, seed: int = SEED
) -> dict[str, float]:
    """Run XGBoost in a subprocess when the main interpreter cannot load libomp safely."""
    helper = REPO / "scripts" / "_xgboost_subprocess.py"
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(
            tmp.name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        npz_path = tmp.name

    last_error = None
    try:
        for py in XGB_PYTHON_CANDIDATES:
            if not Path(py).exists():
                continue
            try:
                proc = subprocess.run(
                    [py, str(helper), npz_path, "--seed", str(seed)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return json.loads(proc.stdout.strip())
            except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as exc:
                last_error = exc
                continue
    finally:
        Path(npz_path).unlink(missing_ok=True)

    raise RuntimeError(f"XGBoost subprocess failed on all Python candidates: {last_error}")


def load_gnn_metrics() -> list[dict]:
    static_path = REPO / "data" / "processed" / "static_gcn_summary.json"
    evolve_path = REPO / "data" / "processed" / "evolvegcn_summary.json"
    rows = []
    for path, name, model_type in [
        (static_path, "Static GCN", "GNN"),
        (evolve_path, "EvolveGCN-H", "GNN"),
    ]:
        if path.exists():
            data = json.loads(path.read_text())
            tm = data["test_metrics"]
            rows.append(
                {
                    "Model": name,
                    "Type": model_type,
                    "AUROC": tm["AUROC"],
                    "F1": tm["F1"],
                    "Precision": tm["Precision"],
                    "Recall": tm["Recall"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading snapshots from {SNAPSHOTS_PATH}")
    snapshots = torch.load(SNAPSHOTS_PATH, map_location="cpu", weights_only=False)

    X_train, y_train = collect_split(snapshots, TRAIN_IDX)
    X_val, y_val = collect_split(snapshots, VAL_IDX)
    X_test, y_test = collect_split(snapshots, TEST_IDX)

    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Features already scaled (StandardScaler on T1-T34 per preprocessing)\n")

    results: list[dict] = []

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight={0: 1.0, 1: CLASS_WEIGHT_ILLICIT},
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    lr_m = metrics_from_arrays(y_test, lr_pred, lr_prob)
    results.append({"Model": "Logistic Regression", "Type": "Tabular", **lr_m})
    print(f"  AUROC={lr_m['AUROC']} F1={lr_m['F1']}\n")

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight={0: 1.0, 1: CLASS_WEIGHT_ILLICIT},
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    rf_m = metrics_from_arrays(y_test, rf_pred, rf_prob)
    results.append({"Model": "Random Forest", "Type": "Tabular", **rf_m})
    print(f"  AUROC={rf_m['AUROC']} F1={rf_m['F1']}\n")

    # 3. XGBoost (isolated subprocess for macOS/Python 3.14 libomp compatibility)
    print("Training XGBoost...")
    xgb_m = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, seed=SEED)
    results.append({"Model": "XGBoost", "Type": "Tabular", **xgb_m})
    print(f"  AUROC={xgb_m['AUROC']} F1={xgb_m['F1']}\n")

    # 4. MLP
    print("Training MLP (165→64→32→2)...")
    mlp = train_mlp(X_train, y_train, X_val, y_val, device)
    mlp_pred, mlp_prob = predict_mlp(mlp, X_test, device)
    mlp_m = metrics_from_arrays(y_test, mlp_pred, mlp_prob)
    results.append({"Model": "MLP", "Type": "Tabular", **mlp_m})
    print(f"  AUROC={mlp_m['AUROC']} F1={mlp_m['F1']}\n")

    # Append GNN results from saved summaries
    gnn_rows = load_gnn_metrics()
    all_rows = results + gnn_rows

    # Sort by AUROC descending
    all_rows.sort(key=lambda r: r["AUROC"], reverse=True)

    # Save JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_rows, indent=2))

    # Save CSV
    csv_path = PAPER_DIR / "baseline_comparison.csv"
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("Model,Type,AUROC,F1,Precision,Recall\n")
        for r in all_rows:
            f.write(
                f"{r['Model']},{r['Type']},{r['AUROC']},{r['F1']},"
                f"{r['Precision']},{r['Recall']}\n"
            )

    # Save LaTeX
    tex_path = PAPER_DIR / "baseline_comparison.tex"
    tex_lines = [
        "% Auto-generated by scripts/run_baseline_models.py",
        "\\begin{table}[htbp]",
        "\\caption{Model Comparison on Elliptic Test Set (T37--T49)}",
        "\\label{tab:baseline_comparison}",
        "\\centering",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Type} & \\textbf{AUROC} & \\textbf{F1} & "
        "\\textbf{Precision} & \\textbf{Recall} \\\\",
        "\\midrule",
    ]
    for r in all_rows:
        name = r["Model"].replace("→", "$\\rightarrow$")
        tex_lines.append(
            f"{name} & {r['Type']} & {r['AUROC']:.4f} & {r['F1']:.4f} & "
            f"{r['Precision']:.4f} & {r['Recall']:.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n")

    print("=" * 60)
    print(f"{'Model':<22} {'AUROC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 60)
    for r in all_rows:
        print(
            f"{r['Model']:<22} {r['AUROC']:>8.4f} {r['F1']:>8.4f} "
            f"{r['Precision']:>8.4f} {r['Recall']:>8.4f}"
        )
    print("=" * 60)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
