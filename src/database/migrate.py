"""Lightweight schema migration helper.

Since the app uses ``Base.metadata.create_all`` (which only creates tables
that don't exist), columns added to SQLAlchemy models after the initial
table creation are never reflected in the real database.  This module runs
safe ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` statements at startup to
bring the schema up to date.

This is **not** a replacement for Alembic — it's a stopgap for dev/staging
environments where Alembic migrations haven't been set up yet.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from .session import async_session_factory

logger = logging.getLogger(__name__)

# Each entry: (table_name, column_ddl)
# column_ddl is the full ``col_name TYPE [CONSTRAINTS]`` fragment.
_MIGRATIONS: list[tuple[str, str]] = [
    (
        "vector_stores",
        "description VARCHAR(512)",
    ),
    (
        "chunks",
        "image_url VARCHAR(1024)",
    ),
]


async def run_schema_migrations() -> None:
    """Execute pending ``ALTER TABLE`` statements for missing columns."""
    async with async_session_factory() as session:
        for table, col_ddl in _MIGRATIONS:
            sql = text(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_ddl}"
            )
            try:
                await session.execute(sql)
                await session.commit()
                logger.debug("migration_applied", table=table, column=col_ddl)
            except Exception as exc:
                # Column likely already exists or table doesn't exist yet —
                # ``create_all`` will handle table creation on next restart.
                logger.debug(
                    "migration_skip",
                    table=table,
                    column=col_ddl,
                    error=str(exc),
                )
                await session.rollback()
    logger.info("schema_migrations_complete")
