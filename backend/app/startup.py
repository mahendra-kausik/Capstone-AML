"""DB init, demo users, ML warmup."""
import logging

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.security import hash_password
from app.database.models import Base, Organization, User, UserRole
from app.database.migrate import upgrade_sqlite_schema
from app.database.session import engine, SessionLocal
from app.ml.loader import load_artifacts
from app.services.drift_service import seed_drift_events

logger = logging.getLogger(__name__)


def init_db():
    Base.metadata.create_all(bind=engine)
    upgrade_sqlite_schema(engine)
    logger.info("Database schema initialized")


def seed_org(db: Session) -> Organization:
    org = db.query(Organization).filter(Organization.slug == "demo-org").first()
    if not org:
        org = Organization(name="Demo Financial Institution", slug="demo-org")
        db.add(org)
        db.flush()
    return org


def seed_users(db: Session):
    settings = get_settings()
    if not settings.seed_demo_users:
        logger.info("Demo user seeding disabled")
        return

    org = seed_org(db)
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
                    org_id=org.id,
                    email_verified=True,
                )
            )
            logger.info("Seeded demo user: %s (%s)", email, role.value)
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
    logger.info("Startup complete")
