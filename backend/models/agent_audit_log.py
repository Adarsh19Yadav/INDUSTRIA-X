"""
backend/models/agent_audit_log.py
==================================
SQLAlchemy ORM model for the agent audit log.

Phase 8: records request metadata and outcome for every agent query.

PRIVACY / SECURITY RULES
-------------------------
- Do NOT store chain-of-thought or hidden reasoning.
- Do NOT store API keys, passwords, or secrets.
- Do NOT store full request/response bodies — only structured metadata.
- response_summary is a short, concise outcome description (max 1 000 chars).
- tools_used is a comma-separated list of tool names (no argument values).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class AgentAuditLog(Base):
    """Immutable audit record for a single agent request."""

    __tablename__ = "agent_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Request identity
    request_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Request content — query only (no hidden reasoning)
    query: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Optional machine context
    machine_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Execution metadata
    tools_used: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    # Concise outcome summary — NOT chain-of-thought
    response_summary: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Observability
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<AgentAuditLog id={self.id} request_id={self.request_id!r} "
            f"status={self.status!r}>"
        )
