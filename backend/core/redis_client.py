"""
backend/core/redis_client.py
============================
Redis connection and basic health check.

Phase 1: connection factory and ping.
Phase 2: sensor stream consumers and publishers will be added.
"""

import redis
from redis import Redis

from backend.core.config import settings

_redis_client: Redis | None = None


def get_redis() -> Redis:
    """
    Return a shared Redis client instance.

    Creates the client on first call (lazy singleton).
    Thread-safe for read operations; the redis-py client is thread-safe.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis_client


def check_redis_connection() -> bool:
    """
    Check whether Redis is reachable.

    Returns True if Redis responds to PING, False otherwise.
    Used by the health check endpoint.
    """
    try:
        client = get_redis()
        return client.ping()
    except Exception:
        return False
