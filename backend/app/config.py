"""Application settings."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AML Intelligence Platform"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Research repo root (models + data); override in Docker as /research
    research_root: Path = Path(__file__).resolve().parents[2]

    # Local default: SQLite (no Docker Postgres required).
    # For Docker/production set DATABASE_URL to PostgreSQL (see .env.example).
    database_url: str = (
        f"sqlite:///{(Path(__file__).resolve().parents[1] / 'aml_platform.db').as_posix()}"
    )

    jwt_secret: str = "change-me-in-production-use-openssl-rand"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    rate_limit_predict: str = "60/minute"
    rate_limit_explain: str = "20/minute"

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    demo_admin_email: str = "admin@example.com"
    demo_admin_password: str = "demoaml2024"
    demo_analyst_email: str = "analyst@example.com"
    demo_analyst_password: str = "demoaml2024"
    demo_viewer_email: str = "viewer@example.com"
    demo_viewer_password: str = "demoaml2024"

    shap_default_nsamples: int = 100
    shap_background_size: int = 50

    @property
    def models_dir(self) -> Path:
        return self.research_root / "models"

    @property
    def processed_dir(self) -> Path:
        return self.research_root / "data" / "processed"

    @property
    def shap_dir(self) -> Path:
        return self.research_root / "data" / "shap"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
