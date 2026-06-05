"""Orchestrate ML inference + DB persistence."""
from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy.orm import Session

from app.database.models import BatchJob, Prediction, Transaction, User
from app.ml.feature_pipeline import (
    features_from_dict,
    features_from_list,
    scale_features,
)
from app.ml.inference import predict, predict_batch
from app.ml.loader import get_artifacts
from app.ml.shap_explainer import generate_shap
from app.schemas.predict import (
    BatchPredictRequest,
    ExplainRequest,
    PredictRequest,
    PredictResponse,
    TopFeature,
)
from app.services import alert_service


def _resolve_features(req: PredictRequest | ExplainRequest) -> np.ndarray:
    if req.features:
        raw = features_from_list(req.features)
    else:
        raw = features_from_dict(req.feature_dict or {})
    artifacts = get_artifacts()
    return scale_features(raw.reshape(1, -1), artifacts.scaler)[0]


def _tops_to_schema(tops: list[dict]) -> list[TopFeature]:
    return [TopFeature(**t) for t in tops]


def _score_fields(out: dict) -> dict:
    """Inference dict may include placeholder top_features; response sets its own."""
    return {k: v for k, v in out.items() if k != "top_features"}


def run_predict(db: Session, user: User | None, req: PredictRequest) -> PredictResponse:
    X = _resolve_features(req)
    edges = None
    out = predict(X, model_name=req.model, tx_id=req.tx_id, edges=edges)
    tops = _fallback_top_from_features(X, out["risk_score"])

    tx = Transaction(
        user_id=user.id if user else None,
        tx_id=req.tx_id,
        time_step=req.time_step,
        features=X.tolist(),
    )
    db.add(tx)
    db.flush()

    pred = Prediction(
        transaction_id=tx.id,
        model_name=req.model,
        risk_score=out["risk_score"],
        prediction=out["prediction"],
        confidence=out["confidence"],
        prob_licit=out["prob_licit"],
        prob_illicit=out["prob_illicit"],
        top_features=[t.model_dump() for t in tops],
    )
    db.add(pred)
    db.flush()
    alert_service.maybe_create_prediction_alert(db, pred, req.tx_id)
    db.commit()

    return PredictResponse(
        tx_id=req.tx_id,
        model=req.model,
        top_features=tops,
        prediction_id=str(pred.id),
        **_score_fields(out),
    )


def _fallback_top_from_features(X: np.ndarray, risk: float) -> list[TopFeature]:
    idx = np.argsort(np.abs(X))[::-1][:5]
    return [
        TopFeature(
            name=f"feat_{int(i)}",
            index=int(i),
            contribution=round(float(abs(X[i]) * risk), 4),
            feature_value=round(float(X[i]), 4),
        )
        for i in idx
    ]


def run_batch(
    db: Session,
    user: User | None,
    req: BatchPredictRequest,
    filename: str | None = None,
) -> dict:
    artifacts = get_artifacts()
    feats_raw, tx_ids, tsteps = [], [], []
    for t in req.transactions:
        if t.features:
            raw = features_from_list(t.features)
        else:
            raw = features_from_dict(t.feature_dict or {})
        feats_raw.append(raw)
        tx_ids.append(t.tx_id)
        tsteps.append(t.time_step)

    X = scale_features(np.stack(feats_raw), artifacts.scaler)
    edge_dicts = [e.model_dump() for e in req.edges] if req.edges else None
    outs = predict_batch(X, tx_ids, model_name=req.model, edges=edge_dicts)

    job = BatchJob(
        user_id=user.id if user else None,
        filename=filename,
        status="completed",
        total_rows=len(tx_ids),
        high_risk_count=sum(1 for o in outs if o["prediction"] == "illicit"),
    )
    db.add(job)
    db.flush()

    results = []
    for i, (tid, o) in enumerate(zip(tx_ids, outs)):
        tx = Transaction(
            user_id=user.id if user else None,
            batch_id=job.id,
            tx_id=tid,
            time_step=tsteps[i],
            features=X[i].tolist(),
        )
        db.add(tx)
        db.flush()
        tops = _fallback_top_from_features(X[i], o["risk_score"])
        pred = Prediction(
            transaction_id=tx.id,
            model_name=req.model,
            risk_score=o["risk_score"],
            prediction=o["prediction"],
            confidence=o["confidence"],
            prob_licit=o["prob_licit"],
            prob_illicit=o["prob_illicit"],
            top_features=[t.model_dump() for t in tops],
        )
        db.add(pred)
        db.flush()
        alert_service.maybe_create_prediction_alert(db, pred, tid)
        results.append(
            PredictResponse(
                tx_id=tid,
                model=req.model,
                top_features=tops,
                prediction_id=str(pred.id),
                **_score_fields(o),
            )
        )

    db.commit()
    mean_risk = float(np.mean([r.risk_score for r in results])) if results else 0.0
    return {
        "job_id": str(job.id),
        "count": len(results),
        "results": results,
        "summary": {
            "high_risk": job.high_risk_count,
            "mean_risk_score": round(mean_risk, 4),
        },
    }


def run_explain(db: Session, user: User | None, req: ExplainRequest) -> dict:
    from app.database.models import ShapResult

    X = _resolve_features(req)
    out = generate_shap(X, model_name=req.model, nsamples=req.nsamples)
    tops = _tops_to_schema(out.get("top_features", []))

    tx = Transaction(
        user_id=user.id if user else None,
        tx_id=req.tx_id,
        time_step=req.time_step,
        features=X.tolist(),
    )
    db.add(tx)
    db.flush()

    pred = Prediction(
        transaction_id=tx.id,
        model_name=req.model,
        risk_score=out["risk_score"],
        prediction=out["prediction"],
        confidence=out.get("confidence", out["risk_score"]),
        prob_licit=out.get("prob_licit", 1 - out["risk_score"]),
        prob_illicit=out.get("prob_illicit", out["risk_score"]),
        top_features=[t.model_dump() for t in tops],
    )
    db.add(pred)
    db.flush()

    shap_row = ShapResult(
        prediction_id=pred.id,
        top_features=[t.model_dump() for t in tops],
        nsamples=req.nsamples,
        method=out.get("method", "kernel_shap"),
    )
    db.add(shap_row)
    alert_service.maybe_create_prediction_alert(db, pred, req.tx_id)
    db.commit()

    return {
        "tx_id": req.tx_id,
        "model": req.model,
        "risk_score": out["risk_score"],
        "prediction": out["prediction"],
        "confidence": out["confidence"],
        "top_features": tops,
        "method": out.get("method", "kernel_shap"),
        "shap_id": str(shap_row.id),
    }
