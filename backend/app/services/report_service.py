"""Report generation — compliance, executive, investigation summaries."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.models import Case, Prediction, Report, Transaction, User
from app.services.drift_service import get_drift_payload


def generate_report(
    db: Session,
    user: User,
    report_type: str,
    title: str | None = None,
    parameters: dict | None = None,
) -> dict:
    params = parameters or {}
    risk_min = float(params.get("risk_min", 0.5))
    limit = min(int(params.get("limit", 200)), 200)

    preds = (
        db.query(Prediction)
        .join(Transaction)
        .filter(Prediction.risk_score >= risk_min)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

    drift = get_drift_payload()
    cases = db.query(Case).count()
    high_risk = sum(1 for p in preds if p.prediction == "illicit")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": report_type,
        "total_scored": db.query(Prediction).count(),
        "high_risk_count": high_risk,
        "open_cases": cases,
        "drift_status": drift.get("summary", {}),
        "model_metrics": {
            "static_gcn_auroc": 0.8573,
            "evolvegcn_auroc": 0.7666,
        },
        "transactions": [
            {
                "tx_id": p.transaction.tx_id if p.transaction else None,
                "risk_score": p.risk_score,
                "prediction": p.prediction,
                "model": p.model_name,
                "confidence": p.confidence,
            }
            for p in preds
        ],
    }

    report = Report(
        user_id=user.id,
        title=title or f"{report_type.replace('_', ' ').title()} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        report_type=report_type,
        parameters=params,
        summary=summary,
        file_format="json",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "id": str(report.id),
        "title": report.title,
        "report_type": report.report_type,
        "summary": summary,
        "created_at": report.created_at.isoformat(),
    }


def export_csv(db: Session, report_type: str = "compliance", risk_min: float = 0.5) -> str:
    preds = (
        db.query(Prediction)
        .join(Transaction)
        .filter(Prediction.risk_score >= risk_min)
        .order_by(Prediction.created_at.desc())
        .limit(200)
        .all()
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["tx_id", "model", "risk_score", "prediction", "confidence", "created_at"])
    for p in preds:
        writer.writerow([
            p.transaction.tx_id if p.transaction else "",
            p.model_name,
            f"{p.risk_score:.4f}",
            p.prediction,
            f"{p.confidence:.4f}",
            p.created_at.isoformat(),
        ])
    return buf.getvalue()


def list_reports(db: Session, limit: int = 20) -> list[dict]:
    rows = db.query(Report).order_by(Report.created_at.desc()).limit(limit).all()
    return [
        {
            "id": str(r.id),
            "title": r.title,
            "report_type": r.report_type,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
