from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.audit import log_action
from app.core.deps import get_current_user, require_roles
from app.core.security import create_access_token, hash_password, verify_password
from app.database.models import User, UserRole
from app.database.session import get_db
from app.schemas.auth import Token, UserCreate, UserLogin, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(body: UserLogin, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(user.email, user.role.value)
    log_action(
        db,
        user,
        "auth.login",
        "users",
        {"email": user.email},
        request.client.host if request.client else None,
    )
    db.commit()
    return Token(access_token=token, role=user.role.value)


@router.post("/register", response_model=UserOut)
def register(
    body: UserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.admin)),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    role = UserRole(body.role) if body.role in UserRole.__members__ else UserRole.analyst
    user = User(email=body.email, hashed_password=hash_password(body.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=str(user.id), email=user.email, role=user.role.value)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=str(user.id), email=user.email, role=user.role.value)
