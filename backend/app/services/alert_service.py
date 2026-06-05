"""Alert engine — generates alerts for high-risk predictions and drift events."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.database.models import Alert, AlertSeverity, AlertStatus, DriftEvent, Prediction


def _severity_from_risk(risk_score: float, prediction: str) -> AlertSeverity | None:
    if prediction == "illicit" and risk_score >= 0.85:
        return AlertSeverity.critical
    if prediction == "illicit" and risk_score >= 0.65:
        return AlertSeverity.high
    if risk_score >= 0.5:
        return AlertSeverity.medium
    return None


def maybe_create_prediction_alert(db: Session, pred: Prediction, tx_id: str) -> Alert | None:
    severity = _severity_from_risk(pred.risk_score, pred.prediction)
    if not severity:
        return None

    existing = (
        db.query(Alert)
        .filter(Alert.prediction_id == pred.id, Alert.alert_type == "high_risk_transaction")
        .first()
    )
    if existing:
        return existing

    label = "Critical Risk" if severity == AlertSeverity.critical else "High Risk"
    alert = Alert(
        prediction_id=pred.id,
        alert_type="high_risk_transaction",
        severity=severity,
        status=AlertStatus.open,
        title=f"{label} Transaction — {tx_id}",
        message=(
            f"Transaction {tx_id} flagged as {pred.prediction} with risk score "
            f"{pred.risk_score:.2%} (model: {pred.model_name})."
        ),
        payload={
            "tx_id": tx_id,
            "risk_score": pred.risk_score,
            "prediction": pred.prediction,
            "model": pred.model_name,
        },
    )
    db.add(alert)
    return alert


def create_drift_alert(db: Session, event: DriftEvent) -> Alert:
    severity = AlertSeverity.critical if event.metric_value < 0.3 else AlertSeverity.high
    alert = Alert(
        drift_event_id=event.id,
        alert_type="model_drift",
        severity=severity,
        status=AlertStatus.open,
        title=f"Model Drift — {event.model_name} ({event.window or 'global'})",
        message=(
            f"Drift detected for {event.model_name}: metric={event.metric_value:.3f}, "
            f"threshold={event.threshold}."
        ),
        payload={"event_type": event.event_type, "metric_value": event.metric_value},
    )
    db.add(alert)
    return alert


def list_alerts(
    db: Session,
    status: str | None = None,
    severity: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    q = db.query(Alert).order_by(Alert.created_at.desc())
    if status and status in AlertStatus.__members__:
        q = q.filter(Alert.status == AlertStatus(status))
    if severity and severity in AlertSeverity.__members__:
        q = q.filter(Alert.severity == AlertSeverity(severity))
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return {
        "items": [_alert_out(a) for a in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "unread_count": db.query(Alert).filter(Alert.status == AlertStatus.open).count(),
    }


def update_alert_status(db: Session, alert_id: str, status: str) -> dict:
    try:
        aid = uuid.UUID(alert_id)
    except ValueError as exc:
        raise ValueError("Invalid alert_id") from exc
    alert = db.query(Alert).filter(Alert.id == aid).first()
    if not alert:
        raise LookupError("Alert not found")
    if status not in AlertStatus.__members__:
        raise ValueError("Invalid status")
    alert.status = AlertStatus(status)
    db.commit()
    db.refresh(alert)
    return _alert_out(alert)


def _alert_out(a: Alert) -> dict:
    return {
        "id": str(a.id),
        "prediction_id": str(a.prediction_id) if a.prediction_id else None,
        "alert_type": a.alert_type,
        "severity": a.severity.value,
        "status": a.status.value,
        "title": a.title,
        "message": a.message,
        "payload": a.payload,
        "created_at": a.created_at.isoformat(),
        "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
    }
