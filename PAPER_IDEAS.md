# AML + Explainable AI — Paper Direction

A plain-language reference for our paper on the **Elliptic Bitcoin dataset** with a focus on
**explainability (XAI)**. Direction 3 is locked and in progress; Direction 1 is the backup if
Direction 3's go/no-go gate fails (see `PROGRESS.md`).

---

## General Terminology

- **Local vs aggregated features** — The features split into **local** (describing the
  transaction itself) and **aggregated** (describing its neighbours in the transaction graph).

- **Concept drift** — When the patterns in the data change over time, so a model trained on old
  data slowly becomes wrong. In Elliptic there is a documented event (a dark-market shutdown,
  called **T43**) where laundering behaviour visibly changed.

- **Static vs temporal model** — A **static** detector is trained once and then fixed. A
  **temporal** detector is designed to keep adapting as time moves forward.

- **Faithfulness** — A check on whether an explanation is *honest*. Test: remove the features
  the explanation calls important — if the model's confidence really drops, the explanation was
  faithful (it reflected what the model actually used).

- **Kendall τ ("tau")** — We use it to check whether a model's
  list of "important features" stays the same or shifts over time (a shift signals drift).

- **Benchmark study** — A fair, head-to-head comparison of several methods that reports which is
  best for what. Reviewers value these because the result is **measured**, not hoped for.

---

## Direction 3 (PRIMARY — locked, in progress)

**1. Title**
Explainable AML under Concept Drift: How Detection Logic Shifts at a Market Shutdown

**2. Description of what is being done**
A benchmark of four graph-based fraud detectors — GCN, GAT, GraphSAGE, and a corrected temporal
model (EvolveGCN-H) — all put through the same SHAP + Kendall-τ drift pipeline. The finding: at
the documented T43 market shutdown, every architecture reorganises which features it relies on (a
shift from transaction-level to neighbourhood features), and the diagnostic detects this
reasoning drift robustly, reporting per-window τ with confidence intervals. Because the shift
appears across all models, it reads as a property of the data rather than a quirk of any one
detector.

**3. What we have done and what needs to be done**
- *Already done:* the Elliptic data pipeline; two graph detectors trained and evaluated (GCN,
  EvolveGCN-H); SHAP explanations per time window; the Kendall-τ drift metric with bootstrap
  confidence intervals and a permutation test; the local-to-aggregated feature-shift result.
  Diagnosed the flaw that handicapped EvolveGCN-H: its GRU weight-evolution step summarised the
  *full* graph (`x.mean(dim=0)`) including the ~77% of nodes that are unlabelled, so the signal
  steering temporal adaptation was dominated by nodes the model never gets supervision on. Fixed
  by summarising over labelled nodes only (`x[label_mask].mean(0)`); message passing still uses
  the full graph (Weber invariant unchanged).
- *Still to do:* add two more graph architectures (GAT and GraphSAGE) and train them; retrain
  EvolveGCN-H with the fix; run all four models through the existing drift-and-explanation
  pipeline; assemble the cross-model comparison (grouped τ chart, local-fraction chart, benchmark
  table, top-10 W4 feature agreement); evaluate the go/no-go gate; write up the "reasoning drift is
  universal" story with supporting statistics.

**4. Chance of publishability and why**
**Solid (~65%).** It reuses the most of our existing graph and drift work, and it is
positive-either-way: either an architecture stays explanation-stable (a useful finding) or none
does (which makes the diagnostic tool the contribution). It scores a little lower than the
deployment-focused benchmarks because a drift-reasoning story is narrower and less immediately
practical, and because it requires training new graph models and re-running the slower explanation
computations.

**Go/no-go gate** (see `PROGRESS.md` for status): holds if all four models flag `τ < 0.70` at
W3→W4, **and** ≥3 of 4 models show top-10 local fraction dropping at W4. If either fails, fall
back to Direction 1 below.

---

## Direction 1 (BACKUP — trigger: Direction 3's go/no-go gate fails)

**1. Title**
Real-Time Explainable Anti–Money-Laundering on Bitcoin

**2. Description of what is being done**
A practical comparison of fraud detectors judged on **three things at once**: how well they
catch laundering (**accuracy** like F1), how fast they run (**latency**), and how trustworthy their
explanations are (**faithfulness**). We compare fast, simple models against the graph model and
produce a clear recommendation for what a bank should actually deploy — a detector that is
accurate **and** fast **and** explainable. We also add a lightweight early-warning signal that
flags when a model's reasoning has drifted, plus a short finding on how detection logic changes
after a market shutdown.

**3. What we have done and what needs to be done**
- *Already done:* the Elliptic data pipeline; trained graph detectors; SHAP explanations; the
  drift metric with statistical rigour; the finding that, after the shutdown, models shift from
  local features to neighbourhood features. (Everything built for Direction 3 — the four-model
  drift benchmark — carries over directly as this paper's drift section.)
- *Still to do:* add fast baseline models (Random Forest, XGBoost, a simple neural network);
  build a latency harness that times each model per transaction; add a faithfulness measurement
  (remove top features and watch confidence drop), using the fast exact explainer for tree models
  and the standard explainer for the rest; add the label-free early-warning signal (run the drift
  check on the model's own predictions instead of true labels); assemble the trade-off analysis
  and write-up.

**4. Chance of publishability and why**
**High (~75%).** Practical comparison studies with a real deployment angle are exactly what
Scopus-indexed venues publish often, and the result is guaranteed because we are *measuring*
quantities rather than hoping a model wins. It also carries several independent findings
(speed–accuracy–trust trade-off, an early-warning signal, and the feature-shift result), so the
paper does not depend on any single one succeeding.

---

## Key References

- **EvolveGCN** — Pareja et al., 2020. The temporal graph-neural-network architecture we use.
- **Elliptic dataset** — Weber et al., 2019. Source of the data; documents the post-shutdown (T43)
  performance collapse that all models exhibit.
- **SHAP** — Lundberg & Lee, 2017. The core explanation method.
- **GNNShap** — Akkas & Azad, 2024. Efficient SHAP for graph neural networks.
- **DBShap** — Shanbhag et al., 2021. Drift detection using Shapley values.
- **SHAP for sensor-drift monitoring** — Cinar et al., 2025. Closest existing methodology to ours.
- **XGBoost + SHAP for AML** — Ertam, 2025. Static explainable AML; notes that temporal drift is
  unresolved (our motivation).
- **EthereumHeist dataset** — Lin et al., 2023. A possible second dataset with *named* (non-
  anonymised) features for a future generalisation study.
