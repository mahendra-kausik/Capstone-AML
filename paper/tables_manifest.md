# Tables Manifest

All numeric values sourced from repository artifacts cited in each table.

## Table I — Dataset Statistics
**Source:** `data/processed/meta.json`, `data/processed/snapshot_stats.csv`

| Statistic | Value |
|-----------|-------|
| Temporal snapshots | 49 (T1–T49) |
| Features per transaction | 165 (94 local + 71 aggregated) |
| Total nodes (processed) | 46,564 |
| Total edges | 36,624 |
| Labelled illicit transactions | 4,545 (9.761%) |
| Train / Val / Test snapshots | T1–T34 / T35–T36 / T37–T49 |
| Class imbalance (train) | 1:8.6 (illicit:licit) |
| Feature scaling | StandardScaler on T1–T34 only |

## Table II — Model Architectures & Hyperparameters
**Source:** `data/processed/static_gcn_summary.json`, `data/processed/evolvegcn_summary.json`, `src/static_gcn/model.py`, `src/evolvegcn/model.py`

| Parameter | Static GCN | EvolveGCN-H |
|-----------|------------|-------------|
| Architecture | 165→64→32→2 (2× GCNConv) | 165→64→32→2 (GRU-evolved weights) |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | 0.0001 |
| Weight decay | 1e-4 | 1e-4 |
| Dropout | 0.5 | 0.5 |
| Class weight (illicit) | 9.0 | 5.0 |
| Max epochs / Patience | 200 / 20 | 500 / 40 |
| Grad clip | — | 1.0 |
| Best validation AUROC | 0.941 | 0.885 |
| Source run ID | — | r025_lr0.0001_cw5.0_h64_do0.5 |

## Table III — Baseline vs. Graph Model Comparison (T37–T49)
**Source:** `paper/baseline_comparison.csv`, `scripts/run_baseline_models.py`, `results/baseline_comparison.json`  
**LaTeX:** `paper/baseline_comparison.tex`

| Model | Type | AUROC | F1 | Precision | Recall |
|-------|------|-------|-----|-----------|--------|
| Random Forest | Tabular | **0.9182** | **0.7585** | **0.8811** | 0.6659 |
| XGBoost | Tabular | 0.9165 | 0.7303 | 0.7906 | 0.6786 |
| Logistic Regression | Tabular | 0.8648 | 0.2902 | 0.1742 | 0.8675 |
| MLP | Tabular | 0.8574 | 0.1787 | 0.0987 | 0.9447 |
| Static GCN | GNN | 0.8573 | 0.4677 | 0.3831 | 0.6002 |
| EvolveGCN-H | GNN | 0.7666 | 0.3269 | 0.2638 | 0.4297 |

## Table IV — GNN Aggregate Test Performance (T37–T49)
**Source:** `data/processed/static_gcn_summary.json`, `data/processed/evolvegcn_summary.json`

| Metric | Static GCN | EvolveGCN-H | Δ (Evolve − Static) |
|--------|------------|-------------|---------------------|
| AUROC | **0.8573** | 0.7666 | −0.0907 |
| F1 | **0.4677** | 0.3269 | −0.1408 |
| Precision | **0.3831** | 0.2638 | −0.1193 |
| Recall | **0.6002** | 0.4297 | −0.1705 |

## Table V — Pre/Post T43 F1 Analysis
**Source:** `data/processed/evolvegcn_summary.json` (Evolve); Static derived from `per_snapshot` in `static_gcn_summary.json` (mean F1 over T37–T42 and T44–T49)

| Model | Pre-T43 Mean F1 | Post-T43 Mean F1 | F1 Drop |
|-------|-----------------|------------------|---------|
| Static GCN | 0.5519 | 0.0446 | 0.5073 |
| EvolveGCN-H | 0.4175 | 0.0493 | 0.3682 |

## Table VI — T43 Snapshot Metrics
**Source:** `data/processed/static_gcn_summary.json`, `data/processed/evolvegcn_summary.json`, `data/processed/snapshot_stats.csv`

| Metric | Static GCN | EvolveGCN-H |
|--------|------------|-------------|
| F1 | 0.0163 | 0.0225 |
| AUROC | 0.6519 | 0.5919 |
| Precision | 0.0101 | 0.0154 |
| Recall | 0.0417 | 0.0417 |
| Illicit nodes at T43 | 24 (1.75% of 1,370 nodes) | 24 |

## Table VII — Kendall τ SHAP Ranking Stability
**Source:** `data/shap/kendall_tau_results.json`, `data/shap/evolvegcn_kendall_tau_results.json`  
**Method:** Top-15 feature union; drift flagged when τ < 0.70

| Window Pair | Static τ (p-value) | Evolve τ (p-value) |
|-------------|-------------------|-------------------|
| W1→W2 | 0.5556 (0.0009) | 0.4386 (0.0083) |
| W2→W3 | 0.5882 (0.0006) | 0.3474 (0.0336) |
| W3→W4 | 0.1474 (0.3859) | **0.1602** (0.3140) |

## Table VIII — Drift Probe (Train T1–T16, Test T33–T49)
**Source:** `drift_probe` fields in summary JSONs

| Model | Pre-T43 Mean F1 | Post-T43 Mean F1 | F1 Drop |
|-------|-----------------|------------------|---------|
| Static GCN | 0.1035 | 0.0117 | 0.0918 |
| EvolveGCN-H | 0.0510 | 0.0461 | **0.0049** |

## Table IX — T43 Detection Audit
**Source:** `data/shap/t43_predictions.csv`, `Notebooks/04_Shap_Drift_Analysis.ipynb`

| Statistic | Value |
|-----------|-------|
| Total nodes at T43 | 1,370 |
| Ground-truth illicit | 24 |
| True positives | 1 |
| False negatives | 23 |
| Detection rate | 4.17% (1/24) |
| Caught tx_id | 92021053 (P(illicit)=0.8483) |

## Table X — SHAP Configuration
**Source:** `Notebooks/04_Shap_Drift_Analysis.ipynb`, `src/evolvegcn/shap_analysis.py`

| Parameter | Value |
|-----------|-------|
| Explainer | KernelSHAP |
| Background samples | 100 licit nodes/window |
| nsamples | 200 |
| Max illicit nodes/window | 1,600 |
| Kendall top-K | 15 |
| Drift threshold | τ < 0.70 |
| Graph mode for SHAP | Tabular (empty edge_index) |
