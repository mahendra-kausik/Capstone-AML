"""Parse and scale 165-dim Elliptic feature vectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_FEATURES = 165
FEATURE_NAMES = [f"feat_{i}" for i in range(NUM_FEATURES)]


def features_from_dict(d: dict) -> np.ndarray:
    vec = np.zeros(NUM_FEATURES, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        if name in d:
            vec[i] = float(d[name])
        elif str(i) in d:
            vec[i] = float(d[str(i)])
    return vec


def features_from_list(lst: list[float]) -> np.ndarray:
    if len(lst) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {len(lst)}")
    return np.asarray(lst, dtype=np.float32)


def parse_csv_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(content))
    df.columns = [c.strip() for c in df.columns]
    return df


def row_to_features(row: pd.Series) -> np.ndarray:
    if all(f in row.index for f in FEATURE_NAMES):
        return row[FEATURE_NAMES].astype(np.float32).values
    # Elliptic raw: col0 tx_id, col1 time_step, rest features
    numeric = row.select_dtypes(include=[np.number])
    if len(numeric) >= NUM_FEATURES:
        return numeric.iloc[:NUM_FEATURES].astype(np.float32).values
    raise ValueError("Row missing feat_0..feat_164 columns")


def scale_features(X: np.ndarray, scaler) -> np.ndarray:
    Xs = scaler.transform(X)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
