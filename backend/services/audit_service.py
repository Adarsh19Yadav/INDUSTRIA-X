"""
backend/services/audit_service.py
====================================
Service layer for agent audit logging.

Phase 8: persists structured metadata for every agent request.

PRIVACY / SECURITY RULES
-------------------------
- Do NOT store API keys, passwords, or secrets.
- Do NOT store chain-of-thought or hidden reasoning.
- Only store concise structured metadata and a short outcome summary.
- Log injection is prevented by sanitising the query before storing
  (newlines and control characters are removed).
- The response_summary is truncated to 1 000 characters maximum.
"""

from __future__ import annotations

import logging
import unicodedata

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.models.agent_audit_log import AgentAuditLog

logger = logging.getLogger(__name__)

# Maximum lengths enforced before DB write
_MAX_QUERY = 1000
_MAX_SUMMARY = 1000
_MAX_TOOLS = 512


def _sanitise_text(text: str | None, max_len: int) -> str | None:
    """Strip control characters that could cause log injection."""
    if text is None:
        return None
    # Remove ASCII control characters (except space)
    cleaned = "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("C") or ch == " "
    )
    return cleaned[:max_len]


def write_audit_log(
    db: Session,
    *,
    request_id: str,
    query: str,
    status: str,
    session_id: str | None = None,
    machine_id: str | None = None,
    tools_used: list[str] | None = None,
    response_summary: str | None = None,
    latency_ms: int | None = None,
) -> AgentAuditLog | None:
    """Write one agent audit log record.

    Returns the created record, or None if the DB write fails gracefully.
    Failures are logged but never raised — audit logging must not break the
    main request path.
    """
    tools_str: str | None = None
    if tools_used:
        tools_str = _sanitise_text(", ".join(tools_used), _MAX_TOOLS)

    record = AgentAuditLog(
        request_id=request_id[:64],
        session_id=(session_id[:64] if session_id else None),
        query=_sanitise_text(query, _MAX_QUERY) or "",
        machine_id=(machine_id[:128] if machine_id else None),
        tools_used=tools_str,
        status=status[:32],
        response_summary=_sanitise_text(response_summary, _MAX_SUMMARY),
        latency_ms=latency_ms,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.debug(
            "[AuditLog] request_id=%s status=%s latency_ms=%s",
            request_id,
            status,
            latency_ms,
        )
        return record
    except SQLAlchemyError as exc:
        db.rollback()
        # Audit failure must never break the caller
        logger.warning("[AuditLog] Failed to write audit log: %s", exc)
        return None
