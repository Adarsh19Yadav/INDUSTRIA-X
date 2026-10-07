"""
backend/core/middleware.py
===========================
Phase 8: request ID middleware and structured logging.

DESIGN
------
- Every request gets a unique request_id (UUID4).
- If the client sends an ``X-Request-ID`` header (valid UUID or alphanumeric),
  that value is preserved and echoed back.
- The request_id is attached to the response as ``X-Request-ID``.
- Structured log lines are emitted for every request:
  endpoint, method, status, latency, request_id.
- Secrets are never logged.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("industria.access")

# Allow: UUIDs, alphanum+hyphen+underscore, max 64 chars
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9\-_]{8,64}$")


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def _validate_client_id(client_id: str | None) -> str | None:
    """Return the client-supplied ID if it is safe, else None."""
    if not client_id:
        return None
    if _SAFE_REQUEST_ID_RE.match(client_id):
        return client_id
    return None


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request_id to every request and add it to the response header."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        client_id = request.headers.get("X-Request-ID")
        request_id = _validate_client_id(client_id) or _generate_request_id()

        # Make it available to downstream handlers
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "request_id=%s method=%s path=%s status=500 latency_ms=%d error=%s",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
                type(exc).__name__,
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def get_request_id(request: Request) -> str:
    """Return the current request_id; generates one if middleware wasn't applied."""
    return getattr(request.state, "request_id", None) or _generate_request_id()
