"""
backend/core/security.py
========================
API key authentication dependency.

Phase 1: simple API key check via request header.
Phase 2+: extend with role-based access if required.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Raises HTTP 403 if the key is missing or incorrect.

    Usage:
        @router.get("/protected", dependencies=[Depends(require_api_key)])
        def protected_endpoint():
            ...
    """
    if api_key is None or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key
