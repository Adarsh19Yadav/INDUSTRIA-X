"""
backend/core/database.py
========================
SQLAlchemy engine, session factory, and base model.

The engine is created lazily (on first use) so that import-time
failures do not prevent the app from starting when the database
is unreachable (e.g. during testing without a live DB).

Phase 1: engine and session configured; tables not yet created.
Phase 2: ORM models and Alembic migrations will be added.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.exc import OperationalError

from backend.core.config import settings


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ── Lazy engine + session ─────────────────────────────────────────────────────
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine() -> Engine:
    """Return the SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=(settings.APP_ENV == "development"),
        )
    return _engine


def _get_session_local() -> sessionmaker:
    """Return the session factory, creating it on first call."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


# ── Dependency ────────────────────────────────────────────────────────────────
def get_db() -> Session:  # type: ignore[return]
    """
    FastAPI dependency that provides a database session.

    Usage:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Check whether the database is reachable.

    Returns True if the DB responds, False otherwise.
    Used by the health check endpoint.
    """
    try:
        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (OperationalError, Exception):
        return False
