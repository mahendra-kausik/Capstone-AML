# UpbitHack (EthereumHeist) processed data

Place preprocessed artifacts here (from `Notebooks/01_UpbitHack_Preprocessing.ipynb`):

- `snapshots.pt` — 49 PyG snapshots, 165 features
- `meta.json`
- `scaler.pkl`
- `snapshot_stats.csv`

**Colab output path:** `My Drive/Capstone/AML Code/data/processed_upbit/`

**Or preprocess locally:**

1. Download [EthereumHeist / UpbitHack](https://www.dropbox.com/scl/fo/ayk5juz7wn5q82o1dlet3/AC8FHG2bjOafiGmGu9W22kc) CSVs into `data/raw/upbit/`
2. `python scripts/preprocess_upbit.py`

**Run validation:**

```bash
export UPBIT_PROCESSED_DIR=/path/to/processed_upbit  # optional
python scripts/run_upbit_validation.py
```

Outputs: `upbit_static_summary.json`, `upbit_evolve_summary.json`, `data/shap/upbit/`, `figures/upbit/`, `results/cross_dataset_results.md`.
