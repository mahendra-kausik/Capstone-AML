#!/usr/bin/env python3
"""Train XGBoost baseline in an isolated subprocess (Python 3.11+ without segfault)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

CLASS_WEIGHT_ILLICIT = 9.0


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=CLASS_WEIGHT_ILLICIT,
        random_state=args.seed,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    print(json.dumps(metrics_from_arrays(y_test, pred, prob)))


if __name__ == "__main__":
    main()
