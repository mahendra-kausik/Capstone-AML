"""Initial schema — all production tables."""

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Schema is managed via SQLAlchemy create_all on startup for compatibility.
    # Run `alembic upgrade head` after setting DATABASE_URL to apply via ORM.
    pass


def downgrade() -> None:
    pass
