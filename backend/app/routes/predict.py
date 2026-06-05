from fastapi import APIRouter, Depends, File, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.core.audit import log_action
from app.core.deps import get_current_user, require_analyst_or_admin
from app.database.models import User
from app.database.session import get_db
from app.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.prediction_service import run_batch, run_explain, run_predict
from app.services.upload_service import process_csv_upload

router = APIRouter(tags=["predict"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
def predict_endpoint(
    request: Request,
    body: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    out = run_predict(db, user, body)
    log_action(
        db,
        user,
        "predict",
        "predictions",
        {"tx_id": body.tx_id, "model": body.model},
        request.client.host if request.client else None,
    )
    db.commit()
    return out


@router.post("/batch-predict", response_model=BatchPredictResponse)
@limiter.limit("30/minute")
def batch_predict(
    request: Request,
    body: BatchPredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    result = run_batch(db, user, body)
    log_action(
        db,
        user,
        "batch_predict",
        "predictions",
        {"count": result["count"], "model": body.model},
        request.client.host if request.client else None,
    )
    db.commit()
    return BatchPredictResponse(**result)


@router.post("/explain", response_model=ExplainResponse)
@limiter.limit("20/minute")
def explain(
    request: Request,
    body: ExplainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    out = run_explain(db, user, body)
    log_action(
        db,
        user,
        "explain",
        "shap",
        {"tx_id": body.tx_id, "model": body.model},
        request.client.host if request.client else None,
    )
    db.commit()
    return ExplainResponse(**out)


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    model: str = "static",
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        from fastapi import HTTPException

        raise HTTPException(413, "File too large (max 10 MB)")
    result = process_csv_upload(db, user, content, file.filename or "upload.csv", model=model)
    log_action(
        db,
        user,
        "upload",
        "batch_jobs",
        {"filename": file.filename, "count": result.get("count"), "model": model},
        request.client.host if request.client else None,
    )
    db.commit()
    return result
