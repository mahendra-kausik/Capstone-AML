import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database.models import BatchJob, Case, Prediction, ShapResult, Transaction, User
from app.database.session import get_db

router = APIRouter(tags=["history"])


def _history_item(p: Prediction, t: Transaction, case: Case | None) -> dict:
    return {
        "prediction_id": str(p.id),
        "tx_id": t.tx_id,
        "time_step": t.time_step,
        "model": p.model_name,
        "risk_score": p.risk_score,
        "prediction": p.prediction,
        "confidence": p.confidence,
        "prob_licit": p.prob_licit,
        "prob_illicit": p.prob_illicit,
        "batch_id": str(t.batch_id) if t.batch_id else None,
        "case_id": str(case.id) if case else None,
        "case_status": case.status.value if case else None,
        "created_at": p.created_at.isoformat(),
    }


@router.get("/history")
def prediction_history(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    risk_min: float | None = None,
    model: str | None = None,
    prediction: str | None = None,
):
    q = (
        db.query(Prediction, Transaction, Case)
        .join(Transaction, Prediction.transaction_id == Transaction.id)
        .outerjoin(Case, Case.prediction_id == Prediction.id)
        .order_by(Prediction.created_at.desc())
    )
    if risk_min is not None:
        q = q.filter(Prediction.risk_score >= risk_min)
    if model:
        q = q.filter(Prediction.model_name == model)
    if prediction:
        q = q.filter(Prediction.prediction == prediction)

    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return {
        "items": [_history_item(p, t, c) for p, t, c in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/history/{prediction_id}")
def prediction_detail(
    prediction_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid prediction_id") from exc

    row = (
        db.query(Prediction, Transaction, Case)
        .join(Transaction, Prediction.transaction_id == Transaction.id)
        .outerjoin(Case, Case.prediction_id == Prediction.id)
        .filter(Prediction.id == pid)
        .first()
    )
    if not row:
        raise HTTPException(404, "Prediction not found")

    p, t, case = row
    shap = (
        db.query(ShapResult)
        .filter(ShapResult.prediction_id == pid)
        .order_by(ShapResult.created_at.desc())
        .first()
    )
    out = _history_item(p, t, case)
    out["top_features"] = p.top_features or []
    out["features"] = t.features
    if shap:
        out["shap"] = {
            "id": str(shap.id),
            "method": shap.method,
            "top_features": shap.top_features,
            "nsamples": shap.nsamples,
            "created_at": shap.created_at.isoformat(),
        }
    return out


@router.get("/batches")
def batch_jobs(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
):
    q = db.query(BatchJob).order_by(BatchJob.created_at.desc())
    rows = q.offset(offset).limit(limit).all()
    return {
        "items": [
            {
                "job_id": str(j.id),
                "filename": j.filename,
                "status": j.status,
                "total_rows": j.total_rows,
                "high_risk_count": j.high_risk_count,
                "created_at": j.created_at.isoformat(),
            }
            for j in rows
        ],
        "limit": limit,
        "offset": offset,
    }
