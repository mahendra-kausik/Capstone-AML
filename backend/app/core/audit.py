"""Audit trail for compliance-sensitive actions."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.database.models import AuditLog, User


def log_action(
    db: Session,
    user: User | None,
    action: str,
    resource: str,
    detail: dict | None = None,
    ip_address: str | None = None,
) -> None:
    db.add(
        AuditLog(
            user_id=user.id if user else None,
            action=action,
            resource=resource,
            detail=detail,
            ip_address=ip_address,
        )
    )
