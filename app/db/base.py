"""Database engine / session management (async SQLAlchemy 2.0).

Works with PostgreSQL (production, asyncpg) and SQLite (tests, aiosqlite).
Schema is created with ``Base.metadata.create_all`` on startup: the schema is
small and single-user, so a migration tool is unnecessary dependency bloat in
V1 (see master prompt section 33).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(url: str) -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
    if url.startswith("postgresql"):
        kwargs.update(pool_size=10, max_overflow=5, pool_pre_ping=True, pool_recycle=1800)
    return kwargs


def init_engine(url: str | None = None) -> AsyncEngine:
    """Create (once) and return the global engine."""
    global _engine, _session_factory
    if _engine is None:
        dsn = url or get_settings().database_url
        _engine = create_async_engine(dsn, **_engine_kwargs(dsn))
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


def get_engine() -> AsyncEngine:
    return init_engine()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    init_engine()
    assert _session_factory is not None
    return _session_factory


async def create_all() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session scope: commit on success, rollback on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
