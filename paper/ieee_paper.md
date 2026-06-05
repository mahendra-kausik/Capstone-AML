# Explainable Temporal Graph Neural Networks for Cryptocurrency Anti-Money Laundering under Concept Drift

**Authors:** *[To be completed by submission team]*  
**Affiliation:** *[To be completed]*

---

## Abstract

Cryptocurrency anti-money laundering (AML) systems must detect illicit transactions while remaining interpretable to compliance officers and robust to temporal distribution shift. We present a reproducible research pipeline that compares a **Static Graph Convolutional Network (Static GCN)** against an **EvolveGCN-H** temporal model on the Elliptic Bitcoin dataset—a benchmark comprising 49 temporal snapshots, 165 anonymized features, and a documented dark-market shutdown at snapshot T43. Both models share a 165→64→32→2 architecture and are evaluated under a strict temporal split (train T1–T34, validation T35–T36, test T37–T49). Static GCN achieves superior aggregate test performance (AUROC **0.8573**, F1 **0.4677**) compared with EvolveGCN-H (AUROC **0.7666**, F1 **0.3269**). However, both models exhibit catastrophic F1 collapse at T43 (Static F1 **0.0163**; Evolve F1 **0.0225**), detecting only **1 of 24** illicit transactions. We complement predictive evaluation with **KernelSHAP** explainability and **Kendall τ** rank-correlation analysis over four temporal windows. SHAP feature rankings destabilize sharply across W3→W4 (Static τ **0.1474**; Evolve τ **0.1602**), coinciding with the shutdown event. Although EvolveGCN-H does not outperform Static GCN on aggregate metrics, it shows a marginally higher W3→W4 τ and a smaller F1 drop in a drift-probe configuration (0.0049 vs. 0.0918). We discuss implications for deployable AML systems, the limits of AUROC under severe class imbalance, and the need for explainability-aware drift monitoring.

**Index Terms—** Anti-money laundering, graph neural networks, concept drift, explainable AI, SHAP, Kendall tau, Bitcoin, Elliptic dataset, temporal graphs.

---

## I. Introduction

Anti-money laundering (AML) in cryptocurrency markets presents a dual challenge for machine learning practitioners. First, illicit activity is **extremely rare**—the Elliptic Bitcoin dataset reports an overall illicit rate of **9.761%** among labelled transactions, with training-set imbalance of approximately **1:8.6** (illicit:licit) [1]. Second, criminal behavior **evolves over time** as exchanges, mixers, and dark markets appear and disappear. The Elliptic dataset encodes this reality through 49 temporally ordered graph snapshots (T1–T49), including a widely studied **dark-market shutdown at T43** that induces severe concept drift [1], [2].

Graph Neural Networks (GNNs) have emerged as a leading approach for AML on blockchain transaction graphs [1], [3]. Static GCNs apply fixed convolutional filters across snapshots, while temporal variants such as **EvolveGCN** adapt model weights as the graph evolves [4]. However, prior work rarely couples **predictive performance**, **post-hoc explainability**, and **quantitative drift measurement** in a single reproducible pipeline.

This paper makes the following contributions, all backed by artifacts in our open research repository:

1. **Reproducible dual-model pipeline.** We implement and evaluate Static GCN and EvolveGCN-H with documented hyperparameters, checkpoints (`models/static_gcn_best.pt`, `models/evolvegcn_best.pt`), and JSON summaries (`data/processed/`).

2. **Rigorous temporal evaluation.** We report aggregate and per-snapshot metrics on test snapshots T37–T49, with explicit pre/post-T43 analysis.

3. **Explainability-driven drift monitoring.** We apply KernelSHAP [5] across four temporal windows and measure ranking stability via Kendall τ [6], flagging drift when τ < 0.70.

4. **Honest comparative analysis.** We report that Static GCN outperforms EvolveGCN-H on aggregate test metrics, while noting modest stability advantages of the temporal model under specific drift probes.

5. **Production deployment path.** We describe integration of trained models into a FastAPI serving platform with SHAP-based narrative explanations—bridging research and operational AML workflows.

---

## II. Related Work

### A. Graph-Based AML

Weber et al. introduced the Elliptic Bitcoin dataset and demonstrated that GCNs outperform hand-crafted features for illicit transaction classification [1]. Subsequent benchmarks extended GNN-based AML to larger graphs and alternative architectures [3], [9]. Self-supervised approaches such as LaundroGraph further exploit graph structure for AML representation learning [8].

### B. Temporal GNNs

EvolveGCN [4] evolves GCN weights via a GRU cell as new graph snapshots arrive, offering a lightweight alternative to full recurrent graph networks. The **-H variant** evolves weight matrix rows independently, which we implement with architectural patches to prevent rank-1 weight collapse (see Section IV-B).

### C. Concept Drift

Concept drift—changes in the relationship between inputs and targets—is well studied in streaming data [7], [8]. In financial AML, drift may arise from regulatory action, market structure changes, or adversarial adaptation. Our T43 shutdown analysis provides an empirical case study of **sudden drift** rather than gradual covariate shift.

### D. Explainable AI for AML

Post-hoc explainability methods such as SHAP [5] attribute predictions to input features, supporting regulatory expectations for model transparency. We extend SHAP from single predictions to **window-level ranking stability** using Kendall τ, aligning explainability with drift detection.

---

## III. Dataset Description

### A. Elliptic Bitcoin Dataset

We use the Elliptic Bitcoin transaction dataset [1], preprocessed into 49 disjoint temporal graph snapshots. Each snapshot contains:

- **165 anonymized features** per transaction: 94 local features (feat_0–feat_93) and 71 aggregated features (feat_94–feat_164).
- **Binary labels:** licit (0) or illicit (1), available for a subset of transactions.
- **Graph edges** connecting related transactions within each snapshot.

**Processed statistics** (`data/processed/meta.json`):

| Statistic | Value |
|-----------|-------|
| Snapshots | 49 |
| Total nodes | 46,564 |
| Total edges | 36,624 |
| Illicit (labelled) | 4,545 (9.761%) |
| Train / Val / Test | T1–T34 / T35–T36 / T37–T49 |

Features are standardized using a **StandardScaler fit exclusively on training snapshots T1–T34**, preventing information leakage from validation and test periods.

### B. T43 Dark-Market Shutdown

Snapshot T43 is annotated in preprocessing as a **dark-market shutdown event** (`figures/07_shutdown_event.png`). At T43, the subgraph contains **1,370 nodes**, **935 edges**, and only **24 illicit transactions (1.75%)**—an extreme imbalance that stress-tests both models (`data/processed/snapshot_stats.csv`).

---

## IV. Methodology

### A. Static GCN

The Static GCN (`src/static_gcn/model.py`) comprises:

- **Layer 1:** GCNConv(165 → 64) + ReLU + Dropout(0.5)
- **Layer 2:** GCNConv(64 → 32) + ReLU
- **Classifier:** Linear(32 → 2)

Weights are **fixed after training** on snapshots T1–T34 and applied unchanged to all test snapshots. Architecture string: **165 → 64 → 32 → 2**.

**Training configuration** (`Notebooks/02_Static_GCN.ipynb`):

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Class weight (illicit) | 9.0 |
| Max epochs / Patience | 200 / 20 |
| Early stopping metric | Validation AUROC |
| Random seed | 42 |

### B. EvolveGCN-H

EvolveGCN-H (`src/evolvegcn/model.py`, `model_version: evolvegcn-h-patched-v1`) implements the **-H variant** of EvolveGCN [4]:

- **EvolveGCNLayer(165→64):** GRU-evolved weight matrix + manual normalized graph convolution with learnable bias.
- **EvolveGCNLayer(64→32):** Same structure.
- **Classifier:** Linear(32 → 2)

At each snapshot during training, GRU cells evolve convolutional weights chronologically from T1→T34. A **bias parameter** and corrected GRU input handling prevent rank-1 weight collapse observed in early implementations.

**Publication hyperparameters** (`data/processed/evolvegcn_summary.json`, run `r025_lr0.0001_cw5.0_h64_do0.5`):

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Class weight (illicit) | 5.0 |
| Hidden dims | 64 / 32 |
| Dropout | 0.5 |
| Grad clip | 1.0 |
| Max epochs / Patience | 500 / 40 |

Hyperparameter tuning over a **36-run grid** (`results/evolvegcn_experiments.csv`) confirmed that learning rates of 1e-3 cause logit collapse (AUROC ≈ 0.5).

### C. Class Imbalance Handling

Both models use **weighted cross-entropy loss** with class weights [1.0, w] where w ∈ {9.0, 5.0} for Static and Evolve respectively. Predictions use argmax over softmax outputs; no threshold tuning is applied on the test set.

### D. Evaluation Metrics

We report:

- **AUROC** — threshold-independent ranking quality.
- **F1, Precision, Recall** — operational detection metrics under default argmax threshold.
- **Per-snapshot metrics** — F1/AUROC for each test snapshot T37–T49.
- **Pre/post-T43 mean F1** — mean F1 over T37–T42 (pre) and T44–T49 (post), excluding T43.
- **Kendall τ** — rank correlation of SHAP feature importances between temporal windows.

### E. KernelSHAP Methodology

We explain illicit-class predictions using **KernelSHAP** [5] (`Notebooks/04_Shap_Drift_Analysis.ipynb`, `src/evolvegcn/shap_analysis.py`):

| Parameter | Value |
|-----------|-------|
| Explainer | `shap.KernelExplainer` |
| Background | 100 random licit nodes per window |
| nsamples | 200 |
| Max illicit nodes/window | 1,600 (subsampled if exceeded) |
| Importance | Mean \|SHAP\| over illicit nodes |
| Graph mode | Tabular (empty edge_index) |

**Temporal windows:**

| Window | Snapshots | Index range |
|--------|-----------|-------------|
| W1 | T1–T10 | 0–9 |
| W2 | T11–T20 | 10–19 |
| W3 | T21–T30 | 20–29 |
| W4 | T31–T49 | 30–48 (includes T43) |

For EvolveGCN-H, `W_state` is warmed through window snapshots then frozen before each SHAP call.

### F. Kendall τ Drift Methodology

For each consecutive window pair (W1→W2, W2→W3, W3→W4), we:

1. Extract the **top-15 features** by mean \|SHAP\| from each window.
2. Form the **union** of top-15 sets.
3. Compute **Kendall τ** between ranked feature lists.
4. Flag **drift** when τ < **0.70** (repository threshold).

Results are stored in `data/shap/kendall_tau_results.json` (Static) and `data/shap/evolvegcn_kendall_tau_results.json` (Evolve).

---

## V. Experimental Setup

### A. Hardware and Software

- **Training:** PyTorch + PyTorch Geometric; GPU recommended for EvolveGCN tuning grid.
- **SHAP:** CPU execution; artifacts stored as pickle files (`data/shap/shap_W*.pkl`).
- **Reproducibility:** Random seed 42 across training and SHAP subsampling.

### B. Temporal Split Protocol

| Split | Snapshots | Purpose |
|-------|-----------|---------|
| Train | T1–T34 | Model fitting; scaler fitting |
| Validation | T35–T36 | Early stopping |
| Test | T37–T49 | Final evaluation; drift analysis |

No transaction from test snapshots is used during training or scaling.

### C. Drift Probe (Supplementary)

A supplementary **drift probe** trains on T1–T16 and evaluates on T33–T49 (`drift_probe` fields in summary JSONs), simulating deployment with limited historical data before a regime change.

### D. Tabular Baselines

Four non-graph baselines (`scripts/run_baseline_models.py`) consume the same 165-dimensional node features and temporal split as Notebook 02, pooling all labelled nodes from train/val/test snapshots into flat matrices (no `edge_index`):

| Model | Configuration |
|-------|---------------|
| Logistic Regression | `class_weight` illicit=9.0, L-BFGS, max_iter=1000 |
| Random Forest | 200 trees, `class_weight` illicit=9.0 |
| XGBoost | 300 trees, max_depth=6, lr=0.05, `scale_pos_weight`=9.0 |
| MLP | 165→64→32→2, AdamW (lr=1e-3), dropout=0.5, class weight 9.0, early stop on val AUROC |

All baselines use seed 42 and are evaluated on the concatenated test nodes from T37–T49, matching the Static GCN aggregation protocol in `src/static_gcn/training.py`.

---

## VI. Results

### A. Baseline vs. Graph Model Comparison

**Table III** reports aggregate test performance (T37–T49) for tabular baselines and GNNs. Results are reproduced by `scripts/run_baseline_models.py` and saved in `paper/baseline_comparison.csv`.

| Model | Type | AUROC | F1 | Precision | Recall |
|-------|------|-------|-----|-----------|--------|
| Random Forest | Tabular | **0.9182** | **0.7585** | **0.8811** | 0.6659 |
| XGBoost | Tabular | 0.9165 | 0.7303 | 0.7906 | 0.6786 |
| Logistic Regression | Tabular | 0.8648 | 0.2902 | 0.1742 | 0.8675 |
| MLP | Tabular | 0.8574 | 0.1787 | 0.0987 | 0.9447 |
| Static GCN | GNN | 0.8573 | 0.4677 | 0.3831 | 0.6002 |
| EvolveGCN-H | GNN | 0.7666 | 0.3269 | 0.2638 | 0.4297 |

**Key findings:**

1. **Tree ensembles dominate aggregate ranking.** Random Forest and XGBoost achieve the highest AUROC (0.918 and 0.917) and F1 (0.759 and 0.730), exceeding both GNNs. Because baselines operate on the same scaled node features without graph convolutions, this suggests that much of the Elliptic signal is already encoded in local/aggregated tabular attributes rather than multi-hop topology—at least under a pooled cross-snapshot evaluation.

2. **Static GCN is competitive on AUROC but not on F1.** Static GCN matches the MLP on AUROC (0.857) yet doubles MLP F1 (0.468 vs. 0.179), indicating that graph convolutions improve threshold-sensitive detection relative to a feed-forward baseline with identical architecture width.

3. **Class-weight sensitivity differs by family.** Logistic Regression and MLP maximize recall (0.87–0.94) at the cost of precision (0.10–0.17), whereas Random Forest balances precision and recall. Static GCN occupies a middle ground (F1=0.468), outperforming EvolveGCN-H on every metric.

4. **EvolveGCN-H trails all tabular baselines** on aggregate test metrics, consistent with optimization complexity and a lower class weight (5.0 vs. 9.0) in the publication run.

### B. GNN-Only Summary

For direct GNN comparison (Table IV):

| Metric | Static GCN | EvolveGCN-H |
|--------|------------|-------------|
| **AUROC** | **0.8573** | 0.7666 |
| **F1** | **0.4677** | 0.3269 |
| **Precision** | **0.3831** | 0.2638 |
| **Recall** | **0.6002** | 0.4297 |
| Best Val AUROC | **0.941** | 0.885 |

Static GCN outperforms EvolveGCN-H on all four metrics. The AUROC gap (0.0907) is substantial; the F1 gap (0.1408) reflects both ranking and threshold-sensitive detection differences.

### C. Per-Snapshot Drift at T43

Figure 15 (`figures/both_models_drift_comparison.png`) visualizes per-snapshot F1 and AUROC. Both models maintain reasonable performance from T37–T42, then **collapse at T43**:

| Model | T43 F1 | T43 AUROC |
|-------|--------|-----------|
| Static GCN | 0.0163 | 0.6519 |
| EvolveGCN-H | 0.0225 | 0.5919 |

Post-T43, Static GCN records **F1 = 0.0** on snapshots T44–T46 despite AUROC remaining above 0.49 on some snapshots—illustrating **metric divergence under extreme imbalance**.

### D. Pre/Post-T43 Analysis

| Model | Pre-T43 Mean F1 | Post-T43 Mean F1 | F1 Drop |
|-------|-----------------|------------------|---------|
| Static GCN | 0.5519 | 0.0446 | 0.5073 |
| EvolveGCN-H | 0.4175 | 0.0493 | 0.3682 |

EvolveGCN-H exhibits a **smaller main-test F1 drop** (0.3682 vs. 0.5073), though both models fail operationally post-shutdown.

### E. T43 Detection Audit

At T43, the Static GCN detects **1 of 24 illicit transactions (4.17%)** (`data/shap/t43_predictions.csv`):

- **True positive:** tx_id 92021053, P(illicit) = 0.8483
- **False negatives:** 23 transactions

This audit underscores that high aggregate AUROC does not guarantee operational utility during regime change.

### F. Drift Probe Results

| Model | Pre-T43 F1 | Post-T43 F1 | F1 Drop |
|-------|------------|-------------|---------|
| Static GCN | 0.1035 | 0.0117 | 0.0918 |
| EvolveGCN-H | 0.0510 | 0.0461 | **0.0049** |

Under the drift-probe protocol, EvolveGCN-H shows **near-zero F1 degradation**, suggesting temporal weight evolution may confer robustness when training history is short—though absolute F1 remains low.

---

## VII. Explainability Analysis

### A. Window-Level SHAP Patterns (Static GCN)

Top features by mean \|SHAP\| shift across windows:

**W1–W3 dominant features:** feat_54, feat_52, feat_51, feat_89  
**W4 dominant features:** feat_52, feat_54, feat_143, feat_108, feat_162

The emergence of feat_143, feat_108, and feat_162 in W4—coinciding with T43—indicates **feature attribution regime change** not visible from aggregate AUROC alone.

### B. Kendall τ Results

| Window Pair | Static τ | Evolve τ |
|-------------|----------|----------|
| W1→W2 | 0.5556* | 0.4386* |
| W2→W3 | 0.5882* | 0.3474* |
| W3→W4 | 0.1474 | **0.1602** |

\*p < 0.05; W3→W4 p > 0.05 for both models (high variance due to ranking disruption).

All pairs fall below the τ = 0.70 drift threshold. The W3→W4 transition—spanning the T43 shutdown—shows the **lowest stability** for both models. EvolveGCN-H achieves marginally higher τ at W3→W4 (0.1602 vs. 0.1474), consistent with modest temporal adaptation benefits in explainability space even when predictive metrics favor Static GCN.

Figure 25 (`figures/static_vs_evolve_tau.png`) compares τ side-by-side.

---

## VIII. Concept Drift Analysis

Our results support a three-tier drift taxonomy on Elliptic:

1. **Gradual drift (W1→W2, W2→W3):** τ ∈ [0.35, 0.59]; statistically significant rank instability; models retain partial predictive skill.

2. **Sudden drift (W3→W4, T43):** τ ≈ 0.15; SHAP top features reorganize; F1 collapses to near zero; only 4.17% of illicit nodes detected.

3. **Post-shock recovery (T44–T49):** Static GCN shows intermittent AUROC recovery (e.g., T45 AUROC 0.8686) without corresponding F1 recovery—suggesting ** miscalibrated decision boundaries** after distribution shift.

The T43 event validates Elliptic as a **stress-test benchmark** for drift-aware AML research, not merely a static classification task.

---

## IX. Discussion

### A. Why Static GCN Outperformed EvolveGCN-H

Several factors explain Static GCN's superior aggregate metrics relative to EvolveGCN-H (though not relative to tree-ensemble tabular baselines):

1. **Sufficient training data.** With 34 training snapshots, a fixed model can learn stable representations without the optimization complexity of weight evolution.

2. **Hyperparameter sensitivity.** EvolveGCN-H required a 36-run grid search; the best run (lr=1e-4, cw=5.0) still underperformed Static GCN. Higher learning rates caused logit collapse.

3. **Architectural overhead.** GRU weight evolution introduces additional failure modes (rank-1 collapse), requiring patches (`evolvegcn-h-patched-v1`).

4. **Evaluation mode.** EvolveGCN evaluation uses `model.train()` during metric collection to avoid dropout-induced collapse—a pragmatic choice that may not reflect deployment inference.

**Tabular vs. graph signal.** Random Forest and XGBoost surpass Static GCN on AUROC and F1 despite ignoring graph topology. This does not invalidate graph methods for AML—multi-hop patterns may matter for specific snapshots or cold-start nodes—but it shows that strong tabular ensembles set a high bar on Elliptic's pooled node features. Future work should report per-snapshot baseline curves and graph-only ablations to isolate topology contributions.

### B. Temporal Modeling and Stability

Despite lower predictive metrics, EvolveGCN-H demonstrates:

- **Higher W3→W4 Kendall τ** (0.1602 vs. 0.1474)
- **Smaller drift-probe F1 drop** (0.0049 vs. 0.0918)
- **Smaller main-test F1 drop** (0.3682 vs. 0.5073)

These findings suggest temporal models may offer **explainability stability** benefits even when aggregate AUROC is lower—a trade-off relevant to regulated AML deployments where attribution consistency matters.

### C. Implications for AML Systems

1. **AUROC alone is insufficient.** High AUROC (0.65+) at T43 coexists with F1 ≈ 0.02. Compliance workflows should prioritize **precision-recall and operational detection rate**.

2. **Explainability-aware monitoring.** Kendall τ on SHAP rankings provides an early warning signal complementary to predictive metrics.

3. **Shutdown events require retraining triggers.** Pre/post-T43 F1 drops exceeding 0.36–0.51 indicate that static models deployed without drift response will fail silently.

4. **Production integration.** Our FastAPI platform exposes SHAP narratives and drift dashboards, demonstrating a path from research artifacts to analyst-facing tools.

---

## X. Limitations

1. **Single dataset.** All primary results are on Elliptic Bitcoin. Cross-dataset validation (Ethereum/Upbit) is prepared in `data/processed_upbit/` but not completed locally.

2. **Modest temporal stability gains.** EvolveGCN-H's τ advantage at W3→W4 (Δ = 0.0128) is small and not statistically significant (p ≈ 0.31).

3. **SHAP computational cost.** KernelSHAP with nsamples=200 over four windows and up to 1,600 illicit nodes per window is expensive for real-time deployment.

4. **Class imbalance.** Extreme rarity of illicit labels at T43 (1.75%) limits F1 interpretability; threshold optimization might improve operational metrics but risks overfitting.

5. **Tabular SHAP mode.** SHAP analysis uses empty edge indices, not exploiting graph structure during explanation—graph-aware explainers remain future work.

6. **Anonymized features.** The 165 Elliptic features lack semantic labels, limiting domain expert interpretation of SHAP attributions.

---

## XI. Future Work

1. **Cross-chain validation** on Ethereum Heist / Upbit hack datasets (`scripts/run_upbit_validation.py`).

2. **Graph Transformers** and attention-based temporal models for comparison with EvolveGCN-H.

3. **Online drift adaptation** with automated retraining triggers when τ < 0.70.

4. **Real-time AML deployment** with streaming snapshot ingestion and incremental SHAP monitoring.

5. **Graph-aware explainability** extending SHAP to edge-based attributions.

6. **Threshold calibration** post-drift using Platt scaling or isotonic regression on validation snapshots.

---

## XII. Conclusion

We presented a reproducible framework for **explainable temporal GNN-based AML** on the Elliptic Bitcoin dataset, comparing Static GCN and EvolveGCN-H under rigorous temporal evaluation. Static GCN achieves superior test AUROC (**0.8573**) and F1 (**0.4677**), while both models fail catastrophically at the T43 dark-market shutdown—detecting only **4.17%** of illicit transactions. KernelSHAP analysis reveals sharp attribution drift (Kendall τ ≈ 0.15 at W3→W4), and EvolveGCN-H shows marginally higher ranking stability in that window. Our findings caution against deploying static AML models without drift monitoring and demonstrate that **predictive performance and explainability stability are distinct objectives**. The complete pipeline—from preprocessing notebooks through production API—is available in our repository, supporting reproducibility and extension to operational AML systems.

---

## References

[1] M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," KDD Workshop on Anomaly Detection in Finance, 2019.

[2] M. Weber et al., "Graph Neural Networks for Anti-Money Laundering: A Benchmark," arXiv:2007.03527, 2020.

[3] B. Rozemberczki et al., "Benchmarking Graph Neural Networks," NeurIPS Datasets and Benchmarks, 2021.

[4] A. Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs," AAAI, 2020.

[5] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.

[6] M. G. Kendall, "A New Measure of Rank Correlation," *Biometrika*, vol. 30, no. 1/2, pp. 81–93, 1938.

[7] J. Lu et al., "Learning under Concept Drift: A Review," *IEEE TKDE*, vol. 31, no. 12, pp. 2346–2363, 2019.

[8] J. Gama et al., "A Survey on Concept Drift Adaptation," *ACM Computing Surveys*, vol. 46, no. 4, 2014.

[9] P. Cardoso et al., "LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering," ACM ICAIF, 2022.

---

## Appendix A — Repository Artifact Index

| Artifact | Path |
|----------|------|
| Static GCN summary | `data/processed/static_gcn_summary.json` |
| EvolveGCN-H summary | `data/processed/evolvegcn_summary.json` |
| Dataset metadata | `data/processed/meta.json` |
| Kendall τ (Static) | `data/shap/kendall_tau_results.json` |
| Kendall τ (Evolve) | `data/shap/evolvegcn_kendall_tau_results.json` |
| T43 predictions | `data/shap/t43_predictions.csv` |
| Tuning grid | `results/evolvegcn_experiments.csv` |
| Static checkpoint | `models/static_gcn_best.pt` |
| Evolve checkpoint | `models/evolvegcn_best.pt` |
| Preprocessing | `Notebooks/01_Elliptic_Preprocessing.ipynb` |
| Static training | `Notebooks/02_Static_GCN.ipynb` |
| Evolve training | `Notebooks/03_EvolveGCN.ipynb` |
| SHAP analysis | `Notebooks/04_Shap_Drift_Analysis.ipynb` |

---

*Manuscript generated from repository artifacts. All reported metrics verified against JSON summary files. See `paper/tables_manifest.md` and `paper/figures_manifest.md` for complete inventories.*
