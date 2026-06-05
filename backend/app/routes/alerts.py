from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database.models import User
from app.database.session import get_db
from app.services import alert_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertStatusUpdate(BaseModel):
    status: str


@router.get("")
def list_alerts(
    status: str | None = None,
    severity: str | None = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return alert_service.list_alerts(db, status=status, severity=severity, limit=limit, offset=offset)


@router.patch("/{alert_id}")
def update_alert(
    alert_id: str,
    body: AlertStatusUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return alert_service.update_alert_status(db, alert_id, body.status)
    except LookupError:
        raise HTTPException(404, "Alert not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
