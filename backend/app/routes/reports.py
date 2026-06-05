from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database.models import User
from app.database.session import get_db
from app.services import report_service

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportRequest(BaseModel):
    report_type: str = "compliance"
    title: str | None = None
    parameters: dict | None = None


@router.post("")
def create_report(
    body: ReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return report_service.generate_report(
        db, user, body.report_type, body.title, body.parameters
    )


@router.get("")
def list_reports(
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return {"items": report_service.list_reports(db, limit=limit)}


@router.get("/export/csv")
def export_csv(
    report_type: str = "compliance",
    risk_min: float = Query(0.5, ge=0, le=1),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    csv_data = report_service.export_csv(db, report_type, risk_min)
    return PlainTextResponse(
        csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{report_type}_report.csv"'},
    )
