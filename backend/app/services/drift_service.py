"""Drift metrics from research SHAP + F1 summaries."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.database.models import DriftEvent
from app.ml.loader import get_artifacts

TAU_ALERT_THRESHOLD = 0.70
T43_STEP = 43


def get_drift_payload() -> dict:
    art = get_artifacts()
    alerts = []

    for row in art.static_tau:
        tau = row.get("tau", 0)
        if tau < TAU_ALERT_THRESHOLD:
            alerts.append(
                {
                    "type": "shap_drift",
                    "model": "static",
                    "window": row.get("comparison"),
                    "tau": tau,
                    "severity": "high" if tau < 0.3 else "medium",
                }
            )

    for row in art.evolve_tau:
        tau = row.get("tau", 0)
        if tau < TAU_ALERT_THRESHOLD:
            alerts.append(
                {
                    "type": "shap_drift",
                    "model": "evolve",
                    "window": row.get("comparison"),
                    "tau": tau,
                    "severity": "high" if tau < 0.3 else "medium",
                }
            )

    static_t43 = _t43_from_summary(art.static_summary)
    evolve_t43 = _t43_from_summary(art.evolve_summary)

    return {
        "kendall_tau": {"static": art.static_tau, "evolve": art.evolve_tau},
        "t43_drift": {"static": static_t43, "evolve": evolve_t43},
        "alerts": alerts,
        "shutdown_step": T43_STEP,
    }


def _t43_from_summary(summary: dict) -> dict:
    if not summary:
        return {}
    per = summary.get("per_snapshot", [])
    pre = [p["F1"] for p in per if p.get("time_step", 0) < T43_STEP]
    post = [p["F1"] for p in per if p.get("time_step", 0) > T43_STEP]
    return {
        "pre_t43_mean_f1": summary.get("pre_t43_mean_f1"),
        "post_t43_mean_f1": summary.get("post_t43_mean_f1"),
        "f1_drop": summary.get("f1_drop"),
        "pre_snapshots": pre,
        "post_snapshots": post,
    }


def seed_drift_events(db: Session) -> None:
    if db.query(DriftEvent).count() > 0:
        return
    payload = get_drift_payload()
    for row in payload["kendall_tau"]["static"]:
        db.add(
            DriftEvent(
                event_type="shap_tau",
                model_name="static",
                window=row.get("comparison"),
                metric_value=float(row.get("tau", 0)),
                threshold=TAU_ALERT_THRESHOLD,
                is_alert=float(row.get("tau", 1)) < TAU_ALERT_THRESHOLD,
                payload=row,
            )
        )
    for row in payload["kendall_tau"]["evolve"]:
        db.add(
            DriftEvent(
                event_type="shap_tau",
                model_name="evolve",
                window=row.get("comparison"),
                metric_value=float(row.get("tau", 0)),
                threshold=TAU_ALERT_THRESHOLD,
                is_alert=float(row.get("tau", 1)) < TAU_ALERT_THRESHOLD,
                payload=row,
            )
        )
    db.commit()
