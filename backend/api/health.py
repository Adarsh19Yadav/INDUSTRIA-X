"""
backend/api/health.py
=====================
Health check endpoints.

GET /api/v1/health  — liveness probe (always 200 if process is running)
GET /api/v1/health/ready — readiness probe (checks DB + Redis)
"""

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter

from backend.core.config import settings
from backend.core.database import check_db_connection
from backend.core.redis_client import check_redis_connection

router = APIRouter()


@router.get("/health", summary="Liveness probe")
def health_liveness() -> dict:
    """
    Liveness probe.

    Returns 200 as long as the process is running.
    Does NOT check downstream dependencies.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": "INDUSTRIA-X",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
        "python": sys.version,
        "platform": platform.system(),
    }


@router.get("/health/ready", summary="Readiness probe")
def health_readiness() -> dict:
    """
    Readiness probe.

    Checks database and Redis connectivity.
    Returns 200 when all dependencies are reachable, 503 otherwise.
    """
    db_ok = check_db_connection()
    redis_ok = check_redis_connection()
    all_ok = db_ok and redis_ok

    return {
        "status": "ready" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database": "ok" if db_ok else "unavailable",
            "redis": "ok" if redis_ok else "unavailable",
        },
    }
