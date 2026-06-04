#!/usr/bin/env python3
"""One-off patch for Notebooks/03_EvolveGCN.ipynb — run from repo root."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "Notebooks" / "03_EvolveGCN.ipynb"

CONFIG_CELL = '''import os
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# ── Repo paths (Colab: mount Drive first, or set CAPSTONE_ROOT) ───────────────
def _find_repo_root():
    candidates = [
        os.environ.get('CAPSTONE_ROOT'),
        '/content/drive/MyDrive/Capstone/AML Code',
        os.path.abspath(os.path.join(os.getcwd(), '..')),
        os.path.abspath('.'),
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'data', 'processed', 'snapshots.pt')):
            return c
    raise FileNotFoundError('Set CAPSTONE_ROOT to Capstone-AML directory')

BASE_DIR = _find_repo_root()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
FIGURES_DIR   = os.path.join(BASE_DIR, 'figures')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')
for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG — single source of truth (matches Static GCN where noted)
# ══════════════════════════════════════════════════════════════════════════════
SEED            = 42
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 40
MAX_EPOCHS      = 500
CLASS_WEIGHT    = 9.0          # illicit class weight (Static GCN uses 9.0)
HIDDEN_DIM      = 64           # layer-1 width; layer-2 = HIDDEN_DIM // 2
DROPOUT         = 0.5
GRAD_CLIP       = 1.0

IN_CHANNELS     = 165
HIDDEN1         = HIDDEN_DIM
HIDDEN2         = HIDDEN_DIM // 2

TRAIN_IDX = list(range(0,  34))
VAL_IDX   = list(range(34, 36))
TEST_IDX  = list(range(36, 49))
ALL_IDX   = list(range(0,  49))
DRIFT_TRAIN_IDX = list(range(0,  16))
DRIFT_TEST_IDX  = list(range(32, 49))
SHUTDOWN_STEP   = 43

RUN_TUNING_GRID = False        # True → run scripts/run_evolvegcn_tuning.py (36 runs)

torch.manual_seed(SEED)
np.random.seed(SEED)

print('Experiment config')
print(f'  BASE_DIR      : {BASE_DIR}')
print(f'  LR            : {LR}')
print(f'  CLASS_WEIGHT  : {CLASS_WEIGHT}')
print(f'  HIDDEN_DIM    : {HIDDEN_DIM} (H2={HIDDEN2})')
print(f'  DROPOUT       : {DROPOUT}')
print(f'  MAX_EPOCHS    : {MAX_EPOCHS}  PATIENCE={PATIENCE}')
print(f'  Train/Val/Test: T1-34 / T35-36 / T37-49')
'''

CELL25_FIX = '''evolve_per_snap = evaluate_per_snapshot(model, ALL_IDX, TEST_IDX)

static_per_snap = static_summary['per_snapshot']
static_map = {r['time_step']: r for r in static_per_snap}

print('PER-SNAPSHOT F1 — Static GCN vs EvolveGCN-H')
print(f'{"T":>4}  {"Static F1":>10}  {"EvolveGCN F1":>13}  {"Δ":>8}  Note')
print('-' * 60)

for r in evolve_per_snap:
    t = r['time_step']
    e_f1 = r['F1']
    s_f1 = static_map.get(t, {}).get('F1', 0.0)
    delta = e_f1 - s_f1
    sign = '+' if delta >= 0 else ''
    note = '← SHUTDOWN' if t == SHUTDOWN_STEP else ''
    print(f'T{t:>2}  {s_f1:>10.4f}  {e_f1:>13.4f}  {sign}{delta:>7.4f}  {note}')

# Pre-T43 = T37–T42; Post-T43 = T44–T49 (exclude shutdown snapshot T43)
e_pre  = [r['F1'] for r in evolve_per_snap if r['time_step'] < SHUTDOWN_STEP]
e_post = [r['F1'] for r in evolve_per_snap if r['time_step'] > SHUTDOWN_STEP]
s_pre  = [r['F1'] for r in static_per_snap if r['time_step'] < SHUTDOWN_STEP]
s_post = [r['F1'] for r in static_per_snap if r['time_step'] > SHUTDOWN_STEP]

print()
print('PRE vs POST T43 SUMMARY  (test snapshots, T43 excluded from means)')
print(f'{"":20}  {"Static GCN":>12}  {"EvolveGCN-H":>13}')
print(f'{"Mean F1 pre-T43":20}  {np.mean(s_pre):>12.4f}  {np.mean(e_pre):>13.4f}')
print(f'{"Mean F1 post-T43":20}  {np.mean(s_post):>12.4f}  {np.mean(e_post):>13.4f}')
print(f'{"Drop":20}  {np.mean(s_pre)-np.mean(s_post):>12.4f}  {np.mean(e_pre)-np.mean(e_post):>13.4f}')
'''

CELL29_FIX = '''s_pre_f1  = float(np.mean([r['F1'] for r in static_per_snap if r['time_step'] < SHUTDOWN_STEP]))
s_post_f1 = float(np.mean([r['F1'] for r in static_per_snap if r['time_step'] > SHUTDOWN_STEP]))
e_pre_f1  = float(np.mean(e_pre))
e_post_f1 = float(np.mean(e_post))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.array([0, 1])
width = 0.3
bars1 = ax.bar(x - width/2, [s_pre_f1, s_post_f1], width, color='#94A3B8', label='Static GCN', alpha=0.9)
bars2 = ax.bar(x + width/2, [e_pre_f1, e_post_f1], width, color='#0D9488', label='EvolveGCN-H', alpha=0.9)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-T43\\n(T37–T42)', 'Post-T43\\n(T44–T49)'], fontsize=12)
ax.set_ylabel('Mean F1 Score (Illicit Class)', fontsize=11)
ax.set_title('Concept Drift: Static GCN vs EvolveGCN-H\\nPre vs Post Dark Market Shutdown (T43)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, max(s_pre_f1, e_pre_f1) * 1.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pre_post_t43_comparison.png'), bbox_inches='tight')
plt.show()
print('Saved → figures/pre_post_t43_comparison.png')
'''

CELL31_FIX = '''print('DRIFT PROBE — Train: T1–T16  |  Test: T33–T49')
print('=' * 55)

torch.manual_seed(SEED)
drift_model = EvolveGCNH(hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT).to(device)
drift_model.reset_weights()
drift_optimizer = torch.optim.AdamW(drift_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
drift_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, CLASS_WEIGHT]).to(device))

for epoch in range(min(MAX_EPOCHS, 200)):
    loss, _ = train_epoch(drift_model, drift_optimizer, drift_criterion, DRIFT_TRAIN_IDX)
    if epoch % 50 == 0:
        print(f'  Epoch {epoch:>3}  Loss: {loss:.4f}')

# Chronological warm-up T1→T49; report T33–T49 (fixes prior bug: all_idx=DRIFT_TEST_IDX only)
probe_results = evaluate_per_snapshot(drift_model, ALL_IDX, DRIFT_TEST_IDX)

print()
print(f'{"T":>4}  {"F1":>8}  {"AUROC":>8}  Note')
print('-' * 40)
for r in probe_results:
    note = '← SHUTDOWN' if r['time_step'] == SHUTDOWN_STEP else ''
    auroc_str = f"{r['AUROC']:.4f}" if not np.isnan(r['AUROC']) else 'nan   '
    print(f'T{r["time_step"]:>2}  {r["F1"]:>8.4f}  {auroc_str:>8}  {note}')

probe_pre  = [r['F1'] for r in probe_results if r['time_step'] < SHUTDOWN_STEP]
probe_post = [r['F1'] for r in probe_results if r['time_step'] > SHUTDOWN_STEP]
static_probe = static_summary.get('drift_probe', {})
print()
print(f'Mean F1 pre-T43  : {np.mean(probe_pre):.4f}  (Static GCN probe: {static_probe.get("mean_f1_pre_t43", "n/a")})')
print(f'Mean F1 post-T43 : {np.mean(probe_post):.4f}  (Static GCN probe: {static_probe.get("mean_f1_post_t43", "n/a")})')
print(f'Drop             : {np.mean(probe_pre) - np.mean(probe_post):.4f}')
'''

CELL33_FIX = '''model.eval()
all_outputs = {}
with torch.no_grad():
    for i in TEST_IDX:
        data = snapshots[i].to(device)
        out, emb = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_outputs[i] = {
            'time_step': snapshots[i].time_step.item(),
            'preds': out.argmax(dim=1).cpu(),
            'probs': probs.cpu(),
            'embeddings': emb.cpu(),
            'y_true': data.y.cpu(),
        }

torch.save(all_outputs, os.path.join(PROCESSED_DIR, 'evolvegcn_outputs.pt'))
print(f'Outputs saved → {os.path.join(PROCESSED_DIR, "evolvegcn_outputs.pt")}')

summary = {
    'model': 'EvolveGCN-H',
    'architecture': f'{IN_CHANNELS} → {HIDDEN1} → {HIDDEN2} → 2  (GRU-evolved)',
    'train_snapshots': 'T1-T34',
    'val_snapshots': 'T35-T36',
    'test_snapshots': 'T37-T49',
    'best_val_auroc': round(float(best_val_auroc), 4),
    'hyperparameters': {
        'LR': LR, 'WEIGHT_DECAY': WEIGHT_DECAY, 'PATIENCE': PATIENCE,
        'MAX_EPOCHS': MAX_EPOCHS, 'CLASS_WEIGHT': CLASS_WEIGHT,
        'HIDDEN_DIM': HIDDEN_DIM, 'DROPOUT': DROPOUT, 'SEED': SEED,
    },
    'test_metrics': {k: round(float(v), 4) for k, v in evolve_metrics.items()},
    'per_snapshot': evolve_per_snap,
    'pre_t43_mean_f1': round(float(np.mean(e_pre)), 4),
    'post_t43_mean_f1': round(float(np.mean(e_post)), 4),
    'f1_drop': round(float(np.mean(e_pre) - np.mean(e_post)), 4),
    'drift_probe': {
        'train': 'T1-T16',
        'test': 'T33-T49',
        'mean_f1_pre_t43': round(float(np.mean(probe_pre)), 4),
        'mean_f1_post_t43': round(float(np.mean(probe_post)), 4),
        'f1_drop': round(float(np.mean(probe_pre) - np.mean(probe_post)), 4),
    },
}

summary_path = os.path.join(PROCESSED_DIR, 'evolvegcn_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved → {summary_path}')
print(f'Checkpoint    → {os.path.join(MODELS_DIR, "evolvegcn_best.pt")}')
'''

CELL15_FIX = '''class_weights = torch.tensor([1.0, CLASS_WEIGHT]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
print(f'Loss      : CrossEntropyLoss  weights=[1.0, {CLASS_WEIGHT}]')
print(f'Optimizer : AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}')
'''

TUNING_CELL = '''# ── Optional: full hyperparameter grid (36 runs) ─────────────────────────────
import subprocess
if RUN_TUNING_GRID:
    script = os.path.join(BASE_DIR, 'scripts', 'run_evolvegcn_tuning.py')
    subprocess.run([sys.executable, script], check=True, cwd=BASE_DIR)
    print('Tuning complete. See results/evolvegcn_experiments.csv and models/evolvegcn_best.pt')
else:
    print('Set RUN_TUNING_GRID=True to run grid search, or execute:')
    print('  python scripts/run_evolvegcn_tuning.py')
'''


def set_cell_src(nb, idx, src):
    nb['cells'][idx]['source'] = [line + '\n' for line in src.split('\n')]
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None


def main():
    with NB_PATH.open() as f:
        nb = json.load(f)

    # Remove standalone Colab drive cell (index 4) — config handles paths
    if 'google.colab' in ''.join(nb['cells'][4].get('source', [])):
        nb['cells'].pop(4)

    set_cell_src(nb, 4, CONFIG_CELL)  # was 5, now 4 after pop
    set_cell_src(nb, 14, CELL15_FIX)
    set_cell_src(nb, 24, CELL25_FIX)
    set_cell_src(nb, 28, CELL29_FIX)
    set_cell_src(nb, 30, CELL31_FIX)
    set_cell_src(nb, 32, CELL33_FIX)

    # Patch model: dropout + train_epoch signature
    model_src = ''.join(nb['cells'][12]['source'])
    model_src = model_src.replace(
        'class EvolveGCNH(nn.Module):\n',
        'class EvolveGCNH(nn.Module):\n',
    )
    model_src = model_src.replace(
        '                 hidden2     = HIDDEN2):\n        super().__init__()\n        self.layer1',
        '                 hidden2     = HIDDEN2,\n                 dropout     = DROPOUT):\n        super().__init__()\n        self.dropout = dropout\n        self.layer1',
    )
    model_src = model_src.replace(
        'x = F.dropout(x, p=0.5, training=self.training)',
        'x = F.dropout(x, p=self.dropout, training=self.training)',
    )
    model_src = model_src.replace(
        'model    = EvolveGCNH().to(device)',
        'model    = EvolveGCNH(hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT).to(device)',
    )
    nb['cells'][12]['source'] = [line + '\n' for line in model_src.split('\n')]

    train_src = ''.join(nb['cells'][16]['source'])
    train_src = train_src.replace(
        'def train_epoch(mdl, optim, idx_list):',
        'def train_epoch(mdl, optim, crit, idx_list):',
    )
    train_src = train_src.replace(
        'loss   = criterion(out, data.y)',
        'loss   = crit(out, data.y)',
    )
    for old, new in [
        ('train_epoch(model, optimizer, TRAIN_IDX)', 'train_epoch(model, optimizer, criterion, TRAIN_IDX)'),
    ]:
        train_src = train_src.replace(old, new)
    loop_src = ''.join(nb['cells'][18]['source'])
    loop_src = loop_src.replace(
        'train_epoch(model, optimizer, TRAIN_IDX)',
        'train_epoch(model, optimizer, criterion, TRAIN_IDX)',
    )
    nb['cells'][18]['source'] = [line + '\n' for line in loop_src.split('\n')]
    nb['cells'][16]['source'] = [line + '\n' for line in train_src.split('\n')]

    loop_src = ''.join(nb['cells'][18]['source'])
    loop_src = loop_src.replace(
        'model = EvolveGCNH().to(device)',
        'model = EvolveGCNH(hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT).to(device)',
    )
    nb['cells'][18]['source'] = [line + '\n' for line in loop_src.split('\n')]

    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['---\n', '## Cell 17 — Hyperparameter Grid (Publication Tuning)\n',
                   'Runs `scripts/run_evolvegcn_tuning.py` → `results/evolvegcn_experiments.csv`'],
        'id': 'tuning-md',
    })
    nb['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'source': [line + '\n' for line in TUNING_CELL.split('\n')],
        'id': 'tuning-code',
        'execution_count': None,
        'outputs': [],
    })

    with NB_PATH.open('w') as f:
        json.dump(nb, f, indent=1)
    print(f'Patched {NB_PATH}')


if __name__ == '__main__':
    main()
