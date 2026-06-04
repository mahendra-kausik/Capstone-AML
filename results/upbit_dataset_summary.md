# UpbitHack (EthereumHeist) — Dataset Audit

**Status:** `MISSING_SNAPSHOTS`
**Processed dir:** `/path/to/processed_upbit`

## Artifact inventory

| File | Present | Size (bytes) |
|------|---------|--------------|
| snapshots.pt | ✗ | — |
| meta.json | ✗ | — |
| scaler.pkl | ✗ | — |
| snapshot_stats.csv | ✗ | — |

## Raw CSVs (for re-preprocessing)

| File | Present |
|------|---------|
| all-tx.csv | ✗ |
| all-address.csv | ✗ |
| accounts-hacker.csv | ✗ |

## Reference (Colab preprocessing run)

| Field | Colab value | Local value |
|-------|-------------|-------------|
| Snapshots | 49 | — |
| Features | 165 | 165 |
| Total nodes | 761,448 | — |
| Total edges | 2,318,459 | — |
| Total illicit | 71,033 | — |
| Illicit % | 9.329 | — |

**Note:** processed_upbit/snapshots.pt not found. Copy from Colab Drive (data/processed_upbit/) or run: python scripts/preprocess_upbit.py after placing CSVs in data/raw/upbit/
