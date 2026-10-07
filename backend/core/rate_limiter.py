"""
backend/core/rate_limiter.py
==============================
Lightweight Redis-backed rate limiting.

Phase 8: protects expensive endpoints (agent/query, assess) from abuse.

DESIGN
------
- Uses a sliding-window counter per (IP or client-id) key.
- Each key has a TTL equal to the window size — keys auto-expire.
- When Redis is unavailable, rate limiting is bypassed gracefully
  (the application remains usable; a warning is logged).
- Limits are read from Settings so they can be configured via env vars.

USAGE (FastAPI dependency)
--------------------------
    from backend.core.rate_limiter import rate_limit_agent

    @router.post("/agent/query", dependencies=[Depends(rate_limit_agent)])
    def agent_query(...): ...
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Depends, HTTPException, Request, status

from backend.core.config import Settings, get_settings
from backend.core.redis_client import get_redis

logger = logging.getLogger(__name__)


def _client_key(request: Request, prefix: str) -> str:
    """Derive a rate-limit key from the request's client IP."""
    client_ip = request.client.host if request.client else "unknown"
    return f"rl:{prefix}:{client_ip}"


def _check_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    """Increment counter; raise 429 if limit exceeded.

    If Redis is unavailable the check is silently skipped.
    """
    try:
        redis = get_redis()
        current = redis.incr(key)
        if current == 1:
            redis.expire(key, window_seconds)
        if current > limit:
            logger.warning("[RateLimit] Key=%s exceeded limit=%d (current=%d)", key, limit, current)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. Maximum {limit} requests "
                    f"per {window_seconds} seconds."
                ),
                headers={"Retry-After": str(window_seconds)},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Redis unavailable — fail open so the app stays usable
        logger.warning("[RateLimit] Redis unavailable, bypassing rate limit: %s", exc)


def make_rate_limit_dependency(prefix: str, limit_attr: str, window_attr: str) -> Callable:
    """Factory that creates a FastAPI dependency for a named rate limit.

    Parameters
    ----------
    prefix:
        String prefix for the Redis key (e.g. "agent").
    limit_attr:
        Attribute name on Settings for the maximum request count.
    window_attr:
        Attribute name on Settings for the window in seconds.
    """

    def _dependency(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> None:
        limit = getattr(settings, limit_attr, 60)
        window = getattr(settings, window_attr, 60)
        key = _client_key(request, prefix)
        _check_rate_limit(key, limit, window)

    return _dependency


# ── Named dependencies ────────────────────────────────────────────────────────

rate_limit_agent = make_rate_limit_dependency(
    prefix="agent",
    limit_attr="RATE_LIMIT_AGENT_PER_MINUTE",
    window_attr="RATE_LIMIT_WINDOW_SECONDS",
)

rate_limit_assess = make_rate_limit_dependency(
    prefix="assess",
    limit_attr="RATE_LIMIT_ASSESS_PER_MINUTE",
    window_attr="RATE_LIMIT_WINDOW_SECONDS",
)
