# Capstone-AML

**Setup for collaborators**

Brief: collaborators must have Git and Git LFS installed to receive the real dataset file `Elliptic Dataset/elliptic_txs_features.csv`. Without LFS they will see a small pointer file.

**Windows (PowerShell)**

```powershell
# Install (Chocolatey) or download installers from upstream sites
choco install git
choco install git-lfs

# Enable LFS and clone
git lfs install
git clone https://github.com/mahendra-kausik/Capstone-AML.git
cd Capstone-AML
git lfs pull

# If you already cloned and see a pointer file:
Get-Content "Elliptic Dataset/elliptic_txs_features.csv" -TotalCount 1
# If it starts with "version https://git-lfs.github.com/spec/v1":
git lfs fetch --all
git lfs checkout

# Python env
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

**macOS (Terminal)**

```bash
# Install Homebrew if needed, then Git + Git LFS
#/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
brew install git-lfs

git lfs install
git clone https://github.com/mahendra-kausik/Capstone-AML.git
cd Capstone-AML
git lfs pull

# If you already cloned and see a pointer file:
head -n 1 "Elliptic Dataset/elliptic_txs_features.csv"
# If it starts with "version https://git-lfs.github.com/spec/v1":
git lfs fetch --all
git lfs checkout

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify & troubleshooting**

- Check LFS environment: `git lfs env`
- List LFS-managed files: `git lfs ls-files`
- Detect pointer file: file starts with `version https://git-lfs.github.com/spec/v1`
- Quota: GitHub LFS storage/bandwidth is billed to the repo owner — downloads/pushes may fail if quota is exceeded.
- If LFS download fails or collaborators cannot use LFS, provide a dataset download (GitHub Release, Google Drive, or S3) and add the link here.

If you need, I can add a Release with the dataset or upload it to a cloud link and update this README.

**EvolveGCN-H tuning (publication pipeline)**

```bash
cd Capstone-AML
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full grid: 36 runs (LR × class_weight × hidden × dropout) — use GPU
python scripts/run_evolvegcn_tuning.py

# Smoke test
python scripts/run_evolvegcn_tuning.py --quick --quick-epochs 3
```

Outputs: `results/evolvegcn_experiments.csv`, `models/evolvegcn_best.pt`, `data/processed/evolvegcn_summary.json`, figures under `figures/`.

**EthereumHeist / UpbitHack validation**

```bash
# 1) Place processed artifacts (see data/processed_upbit/README.md)
#    or: raw CSVs in data/raw/upbit/ then:
python scripts/preprocess_upbit.py

# 2) Full pipeline (audit → train → SHAP → paper tables)
python scripts/run_upbit_validation.py

# Optional: external processed dir
export UPBIT_PROCESSED_DIR="/path/to/processed_upbit"
python scripts/run_upbit_validation.py
```

Outputs: `data/processed_upbit/upbit_static_summary.json`, `upbit_evolve_summary.json`, `data/shap/upbit/`, `figures/upbit/`, `results/cross_dataset_results.md`, `results/ieee_results_section_draft.md`. Elliptic files are not modified.

**AML Intelligence Platform (full-stack)**

Production app: `backend/` (FastAPI) + `frontend/` (Next.js 15). See `docs/platform/DEPLOYMENT.md`.

```bash
docker compose up --build
# Login: analyst@aml.local / demoaml2024
# Upload: samples/demo_transactions.csv
```

---

## Publication Reproduction (Journal Experiments)

Reproduces all Phase A experiments for `paper/journal_manuscript.md` without retraining GNNs.

**Prerequisites:** `data/processed/snapshots.pt`, `models/static_gcn_best.pt`, `models/evolvegcn_best.pt`, `data/shap/kendall_tau_results.json`

```bash
cd Capstone-AML
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-command pipeline (baselines, ablation, early warning, stats, figures)
chmod +x reproduce.sh
./reproduce.sh

# Or run individual experiments:
python scripts/run_graph_ablation.py
python scripts/run_per_snapshot_metrics.py
python scripts/early_warning_analysis.py
python scripts/run_calibration.py
python scripts/run_statistical_validation.py
python scripts/generate_publication_figures.py
```

**Outputs:** `results/*.csv`, `figures/*.png`, `paper/figures/*.png`

**Status report:** `PUBLICATION_READINESS_REPORT.md`

**Cross-dataset (Upbit):** See `docs/UPBIT_DATA_ACQUISITION.md` — required before journal submission.
# AML_Intelligence
