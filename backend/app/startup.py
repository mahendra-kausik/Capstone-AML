"""DB init, demo users, ML warmup."""
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.security import hash_password
from app.database.models import Base, User, UserRole
from app.database.session import engine, SessionLocal
from app.ml.loader import load_artifacts
from app.services.drift_service import seed_drift_events


def init_db():
    Base.metadata.create_all(bind=engine)


def seed_users(db: Session):
    settings = get_settings()
    demos = [
        (settings.demo_admin_email, settings.demo_admin_password, UserRole.admin),
        (settings.demo_analyst_email, settings.demo_analyst_password, UserRole.analyst),
        (settings.demo_viewer_email, settings.demo_viewer_password, UserRole.viewer),
    ]
    for email, password, role in demos:
        if not db.query(User).filter(User.email == email).first():
            db.add(
                User(
                    email=email,
                    hashed_password=hash_password(password),
                    role=role,
                )
            )
    db.commit()


def on_startup():
    init_db()
    load_artifacts(get_settings())
    db = SessionLocal()
    try:
        seed_users(db)
        seed_drift_events(db)
    finally:
        db.close()
