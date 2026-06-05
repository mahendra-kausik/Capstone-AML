"""CSV upload → batch predict."""
from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy.orm import Session

from app.database.models import User
from app.ml.feature_pipeline import parse_csv_bytes, row_to_features
from app.schemas.predict import BatchPredictRequest, TransactionInput
from app.services.prediction_service import run_batch


def process_csv_upload(
    db: Session,
    user: User | None,
    content: bytes,
    filename: str,
    model: str = "static",
) -> dict:
    df = parse_csv_bytes(content)
    transactions = []
    for _, row in df.iterrows():
        raw_id = row.get("tx_id", row.iloc[0] if len(row) else uuid.uuid4())
        try:
            tx_id = str(int(float(raw_id)))
        except (TypeError, ValueError):
            tx_id = str(raw_id)
        time_step = int(row["time_step"]) if "time_step" in row and not np.isnan(row["time_step"]) else None
        raw = row_to_features(row)
        transactions.append(
            TransactionInput(
                tx_id=tx_id,
                time_step=time_step,
                features=raw.tolist(),
            )
        )

    req = BatchPredictRequest(model=model, transactions=transactions)
    result = run_batch(db, user, req, filename=filename)
    result["filename"] = filename
    return result
