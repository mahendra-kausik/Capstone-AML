from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_roles
from app.database.models import AuditLog, User, UserRole
from app.database.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit-logs")
def audit_logs(
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.admin)),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    rows = q.offset(offset).limit(limit).all()
    return {
        "items": [
            {
                "id": str(r.id),
                "action": r.action,
                "resource": r.resource,
                "detail": r.detail,
                "user_email": r.user.email if r.user else None,
                "ip_address": r.ip_address,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.admin)),
):
    users = db.query(User).order_by(User.created_at).all()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "role": u.role.value,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]
