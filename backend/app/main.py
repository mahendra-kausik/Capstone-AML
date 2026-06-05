"""AML Intelligence Platform API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings
from app.routes import admin, auth, cases, drift, graph, history, metrics, predict
from app.routes.predict import limiter
from app.startup import on_startup

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Temporal GNN AML serving — Static GCN & EvolveGCN-H",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = FastAPI()
prefix = settings.api_prefix

app.include_router(auth.router, prefix=prefix)
app.include_router(predict.router, prefix=prefix)
app.include_router(metrics.router, prefix=prefix)
app.include_router(drift.router, prefix=prefix)
app.include_router(history.router, prefix=prefix)
app.include_router(cases.router, prefix=prefix)
app.include_router(graph.router, prefix=prefix)
app.include_router(admin.router, prefix=prefix)


@app.on_event("startup")
def startup():
    on_startup()


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "api": settings.api_prefix,
    }
