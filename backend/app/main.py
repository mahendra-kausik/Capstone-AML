"""AML Intelligence Platform API."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.core.logging_config import setup_logging
from app.database.session import SessionLocal
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.routes import admin, alerts, auth, cases, drift, graph, history, metrics, predict, reports
from app.routes.auth import limiter as auth_limiter
from app.routes.predict import limiter as predict_limiter
from app.startup import on_startup

settings = get_settings()
setup_logging(settings.debug)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Temporal GNN AML serving — Static GCN & EvolveGCN-H",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.state.limiter = predict_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = settings.api_prefix

app.include_router(auth.router, prefix=prefix)
app.include_router(predict.router, prefix=prefix)
app.include_router(metrics.router, prefix=prefix)
app.include_router(drift.router, prefix=prefix)
app.include_router(history.router, prefix=prefix)
app.include_router(cases.router, prefix=prefix)
app.include_router(graph.router, prefix=prefix)
app.include_router(admin.router, prefix=prefix)
app.include_router(alerts.router, prefix=prefix)
app.include_router(reports.router, prefix=prefix)


@app.on_event("startup")
def startup():
    on_startup()


@app.get("/health")
def health():
    checks: dict = {"status": "ok", "service": settings.app_name, "checks": {}}
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["checks"]["database"] = "ok"
    except Exception as e:
        checks["status"] = "degraded"
        checks["checks"]["database"] = f"error: {e}"

    try:
        from app.ml.loader import get_artifacts

        get_artifacts()
        checks["checks"]["ml_models"] = "ok"
    except Exception as e:
        checks["status"] = "degraded"
        checks["checks"]["ml_models"] = f"error: {e}"

    return checks


@app.get("/metrics")
def prometheus_metrics():
    """Basic metrics endpoint for monitoring."""
    from app.database.models import Alert, AlertStatus, Case, Prediction

    db = SessionLocal()
    try:
        return {
            "predictions_total": db.query(Prediction).count(),
            "cases_total": db.query(Case).count(),
            "alerts_open": db.query(Alert).filter(Alert.status == AlertStatus.open).count(),
        }
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs" if settings.debug else None,
        "api": settings.api_prefix,
        "health": "/health",
    }
