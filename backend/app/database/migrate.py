"""Lightweight schema upgrades for local SQLite (create_all does not alter existing tables)."""
import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _column_exists(engine: Engine, table: str, column: str) -> bool:
    insp = inspect(engine)
    if table not in insp.get_table_names():
        return False
    return column in {c["name"] for c in insp.get_columns(table)}


def _table_exists(engine: Engine, table: str) -> bool:
    return table in inspect(engine).get_table_names()


def upgrade_sqlite_schema(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        if _table_exists(engine, "users") and not _column_exists(engine, "users", "org_id"):
            logger.warning("Upgrading SQLite schema — adding missing columns")
            for stmt in (
                "ALTER TABLE users ADD COLUMN org_id CHAR(32)",
                "ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0 NOT NULL",
            ):
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    logger.debug("Schema statement skipped: %s (%s)", stmt, e)

        if _table_exists(engine, "reports") and not _column_exists(engine, "reports", "file_format"):
            try:
                conn.execute(text("ALTER TABLE reports ADD COLUMN file_format VARCHAR(16) DEFAULT 'json'"))
            except Exception:
                pass

    logger.info("SQLite schema upgrade check complete")
