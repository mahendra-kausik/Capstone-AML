from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.audit import log_action
from app.core.deps import get_current_user, require_analyst_or_admin
from app.database.models import User
from app.database.session import get_db
from app.services.case_service import add_note, create_case, get_case, list_cases, update_case

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseCreate(BaseModel):
    prediction_id: str
    title: str | None = None
    priority: str = "medium"


class CaseUpdate(BaseModel):
    status: str | None = None
    priority: str | None = None
    assignee_id: str | None = None
    title: str | None = None


class NoteCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


@router.get("")
def cases_list(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
    status: str | None = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    return list_cases(db, status=status, limit=limit, offset=offset)


@router.post("")
def cases_create(
    body: CaseCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    out = create_case(db, user, body.prediction_id, body.title, body.priority)
    log_action(db, user, "case.create", "cases", {"case_id": out["id"]}, request.client.host if request.client else None)
    db.commit()
    return out


@router.get("/{case_id}")
def cases_detail(case_id: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return get_case(db, case_id)


@router.patch("/{case_id}")
def cases_update(
    case_id: str,
    body: CaseUpdate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    out = update_case(
        db,
        case_id,
        status=body.status,
        priority=body.priority,
        assignee_id=body.assignee_id,
        title=body.title,
    )
    log_action(db, user, "case.update", "cases", {"case_id": case_id}, request.client.host if request.client else None)
    db.commit()
    return out


@router.post("/{case_id}/notes")
def cases_add_note(
    case_id: str,
    body: NoteCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst_or_admin),
):
    out = add_note(db, user, case_id, body.content)
    log_action(db, user, "case.note", "cases", {"case_id": case_id}, request.client.host if request.client else None)
    db.commit()
    return out
