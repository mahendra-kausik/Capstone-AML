"""Default paths and split indices for EvolveGCN-H experiments."""
from pathlib import Path

# Repo root: Capstone-AML/
REPO_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

SNAPSHOTS_PATH = PROCESSED_DIR / "snapshots.pt"
STATIC_SUMMARY_PATH = PROCESSED_DIR / "static_gcn_summary.json"
EVOLVE_SUMMARY_PATH = PROCESSED_DIR / "evolvegcn_summary.json"
EXPERIMENTS_CSV = RESULTS_DIR / "evolvegcn_experiments.csv"
BEST_CHECKPOINT = MODELS_DIR / "evolvegcn_best.pt"
BEST_HISTORY_PATH = RESULTS_DIR / "evolvegcn_best_history.json"

TRAIN_IDX = list(range(0, 34))
VAL_IDX = list(range(34, 36))
TEST_IDX = list(range(36, 49))
ALL_IDX = list(range(0, 49))
DRIFT_TRAIN_IDX = list(range(0, 16))
DRIFT_TEST_IDX = list(range(32, 49))

SHUTDOWN_STEP = 43
IN_CHANNELS = 165

# Hyperparameter search grid (Steps 3–4)
# 1e-3 causes logit collapse (AUROC=0.5) for most EvolveGCN-H runs on Elliptic
LR_GRID = [5e-4, 1e-4]
CLASS_WEIGHT_GRID = [5.0, 7.0, 9.0]
HIDDEN_DIM_GRID = [64, 128]
DROPOUT_GRID = [0.3, 0.5]

# Fixed training defaults (fair comparison with Static GCN)
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 40
GRAD_CLIP = 1.0
SEED = 42
