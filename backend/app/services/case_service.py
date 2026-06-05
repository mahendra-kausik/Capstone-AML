"""Investigation case workflow tied to predictions."""
from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.database.models import Case, CaseNote, CasePriority, CaseStatus, Prediction, User


def _case_out(case: Case, db: Session) -> dict:
    pred = case.prediction
    tx = pred.transaction if pred else None
    assignee_email = None
    if case.assignee_id:
        u = db.query(User).filter(User.id == case.assignee_id).first()
        assignee_email = u.email if u else None
    return {
        "id": str(case.id),
        "prediction_id": str(case.prediction_id),
        "title": case.title,
        "status": case.status.value,
        "priority": case.priority.value,
        "assignee_id": str(case.assignee_id) if case.assignee_id else None,
        "assignee_email": assignee_email,
        "tx_id": tx.tx_id if tx else None,
        "risk_score": pred.risk_score if pred else None,
        "prediction": pred.prediction if pred else None,
        "model": pred.model_name if pred else None,
        "created_at": case.created_at.isoformat(),
        "updated_at": case.updated_at.isoformat(),
        "notes_count": len(case.notes),
    }


def create_case(
    db: Session,
    user: User,
    prediction_id: str,
    title: str | None = None,
    priority: str = "medium",
) -> dict:
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid prediction_id") from exc

    pred = db.query(Prediction).filter(Prediction.id == pid).first()
    if not pred:
        raise HTTPException(404, "Prediction not found")
    if db.query(Case).filter(Case.prediction_id == pid).first():
        raise HTTPException(409, "Case already exists for this prediction")

    tx = pred.transaction
    prio = CasePriority(priority) if priority in CasePriority.__members__ else CasePriority.medium
    case = Case(
        prediction_id=pid,
        title=title or f"Alert — {tx.tx_id if tx else prediction_id}",
        priority=prio,
        assignee_id=user.id,
        created_by_id=user.id,
        status=CaseStatus.open,
    )
    db.add(case)
    db.flush()
    db.refresh(case)
    return _case_out(case, db)


def list_cases(
    db: Session,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    q = db.query(Case).order_by(Case.updated_at.desc())
    if status and status in CaseStatus.__members__:
        q = q.filter(Case.status == CaseStatus(status))
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return {
        "items": [_case_out(c, db) for c in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_case(db: Session, case_id: str) -> dict:
    try:
        cid = uuid.UUID(case_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid case_id") from exc
    case = db.query(Case).filter(Case.id == cid).first()
    if not case:
        raise HTTPException(404, "Case not found")
    out = _case_out(case, db)
    out["notes"] = [
        {
            "id": str(n.id),
            "content": n.content,
            "author_email": n.author.email if n.author else None,
            "created_at": n.created_at.isoformat(),
        }
        for n in case.notes
    ]
    return out


def update_case(
    db: Session,
    case_id: str,
    status: str | None = None,
    priority: str | None = None,
    assignee_id: str | None = None,
    title: str | None = None,
) -> dict:
    try:
        cid = uuid.UUID(case_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid case_id") from exc
    case = db.query(Case).filter(Case.id == cid).first()
    if not case:
        raise HTTPException(404, "Case not found")

    if status and status in CaseStatus.__members__:
        case.status = CaseStatus(status)
    if priority and priority in CasePriority.__members__:
        case.priority = CasePriority(priority)
    if assignee_id:
        try:
            case.assignee_id = uuid.UUID(assignee_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid assignee_id") from exc
    if title:
        case.title = title

    db.commit()
    db.refresh(case)
    return _case_out(case, db)


def add_note(db: Session, user: User, case_id: str, content: str) -> dict:
    try:
        cid = uuid.UUID(case_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid case_id") from exc
    case = db.query(Case).filter(Case.id == cid).first()
    if not case:
        raise HTTPException(404, "Case not found")

    note = CaseNote(case_id=cid, author_id=user.id, content=content.strip())
    db.add(note)
    db.commit()
    db.refresh(note)
    return {
        "id": str(note.id),
        "content": note.content,
        "author_email": user.email,
        "created_at": note.created_at.isoformat(),
    }
