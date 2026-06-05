from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database.models import DriftEvent, User
from app.database.session import get_db
from app.services.drift_service import get_drift_payload

router = APIRouter(tags=["drift"])


@router.get("/drift")
def drift_dashboard(_: User = Depends(get_current_user)):
    return get_drift_payload()


@router.get("/drift/events")
def drift_events(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    rows = db.query(DriftEvent).order_by(DriftEvent.detected_at.desc()).limit(50).all()
    return [
        {
            "id": str(r.id),
            "event_type": r.event_type,
            "model_name": r.model_name,
            "window": r.window,
            "metric_value": r.metric_value,
            "is_alert": r.is_alert,
            "detected_at": r.detected_at.isoformat(),
        }
        for r in rows
    ]
