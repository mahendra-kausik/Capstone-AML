"""Paths and split indices for EthereumHeist (UpbitHack) validation."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Override with UPBIT_PROCESSED_DIR=/path/to/processed_upbit if artifacts live elsewhere
_env_processed = os.environ.get("UPBIT_PROCESSED_DIR")
PROCESSED_DIR = Path(_env_processed) if _env_processed else REPO_ROOT / "data" / "processed_upbit"
RAW_DIR = Path(os.environ.get("UPBIT_RAW_DIR", REPO_ROOT / "data" / "raw" / "upbit"))

MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures" / "upbit"
RESULTS_DIR = REPO_ROOT / "results"
SHAP_DIR = REPO_ROOT / "data" / "shap" / "upbit"

SNAPSHOTS_PATH = PROCESSED_DIR / "snapshots.pt"
META_PATH = PROCESSED_DIR / "meta.json"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
STATS_PATH = PROCESSED_DIR / "snapshot_stats.csv"

STATIC_SUMMARY_PATH = PROCESSED_DIR / "upbit_static_summary.json"
EVOLVE_SUMMARY_PATH = PROCESSED_DIR / "upbit_evolve_summary.json"
AUDIT_PATH = RESULTS_DIR / "upbit_dataset_audit.json"
AUDIT_TABLE_PATH = RESULTS_DIR / "upbit_dataset_summary.md"

STATIC_CHECKPOINT = MODELS_DIR / "upbit_static_gcn_best.pt"
EVOLVE_CHECKPOINT = MODELS_DIR / "upbit_evolvegcn_best.pt"

# Elliptic reference (read-only; do not overwrite)
ELLIPTIC_STATIC_SUMMARY = REPO_ROOT / "data" / "processed" / "static_gcn_summary.json"
ELLIPTIC_EVOLVE_SUMMARY = REPO_ROOT / "data" / "processed" / "evolvegcn_summary.json"
ELLIPTIC_STATIC_TAU = REPO_ROOT / "data" / "shap" / "kendall_tau_results.json"
ELLIPTIC_EVOLVE_TAU = REPO_ROOT / "data" / "shap" / "evolvegcn_kendall_tau_results.json"

TRAIN_IDX = list(range(0, 34))
VAL_IDX = list(range(34, 36))
TEST_IDX = list(range(36, 49))
ALL_IDX = list(range(0, 49))
DRIFT_TRAIN_IDX = list(range(0, 16))
DRIFT_TEST_IDX = list(range(32, 49))

IN_CHANNELS = 165
SHUTDOWN_STEP = 43

# Match Elliptic Static GCN notebook
STATIC_LR = 0.001
STATIC_CLASS_WEIGHT = 9.0
STATIC_HIDDEN1 = 64
STATIC_HIDDEN2 = 32
STATIC_DROPOUT = 0.5
STATIC_MAX_EPOCHS = 200
STATIC_PATIENCE = 40
STATIC_WEIGHT_DECAY = 1e-4

# Patched EvolveGCN-H publication config (Elliptic best non-collapsed run)
EVOLVE_LR = 1e-4
EVOLVE_CLASS_WEIGHT = 5.0
EVOLVE_DROPOUT = 0.5
EVOLVE_MAX_EPOCHS = 500
EVOLVE_PATIENCE = 40
SEED = 42
