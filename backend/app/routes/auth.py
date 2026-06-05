import hashlib
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.audit import log_action
from app.core.deps import get_current_user, require_roles
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.database.models import RefreshToken, User, UserRole
from app.database.session import get_db
from app.routes.predict import limiter
from app.schemas.auth import RefreshRequest, Token, UserCreate, UserLogin, UserOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(db: Session, user: User) -> Token:
    access = create_access_token(user.email, user.role.value)
    raw_refresh, token_hash, expires_at = create_refresh_token(user.email)
    db.add(RefreshToken(user_id=user.id, token_hash=token_hash, expires_at=expires_at))
    db.flush()
    return Token(access_token=access, refresh_token=raw_refresh, role=user.role.value)


@router.post("/login", response_model=Token)
@limiter.limit(get_settings().rate_limit_login)
def login(body: UserLogin, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        logger.warning("Failed login attempt for %s", body.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account disabled")
    token = _issue_tokens(db, user)
    log_action(
        db,
        user,
        "auth.login",
        "users",
        {"email": user.email},
        request.client.host if request.client else None,
    )
    db.commit()
    logger.info("User logged in: %s", user.email)
    return token


@router.post("/refresh", response_model=Token)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    row = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == token_hash, RefreshToken.revoked.is_(False))
        .first()
    )
    if not row:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    from datetime import datetime, timezone

    if row.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")
    user = db.query(User).filter(User.id == row.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")
    row.revoked = True
    token = _issue_tokens(db, user)
    db.commit()
    return token


@router.post("/logout")
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if row:
        row.revoked = True
        db.commit()
    return {"status": "ok"}


@router.post("/register", response_model=UserOut)
def register(
    body: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(UserRole.admin)),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    role = UserRole(body.role) if body.role in UserRole.__members__ else UserRole.analyst
    user = User(email=body.email, hashed_password=hash_password(body.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    log_action(db, admin, "auth.register", "users", {"email": user.email, "role": role.value})
    db.commit()
    return UserOut(id=str(user.id), email=user.email, role=user.role.value)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=str(user.id), email=user.email, role=user.role.value)
