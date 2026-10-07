"""backend/api/agent.py — Phase 7/8 Agent API Router.

Endpoint
--------
POST /api/v1/agent/query
    Accepts a natural-language query and optional machine sensor readings.
    Returns a structured decision-support response with evidence and sources.

Phase 8 additions
-----------------
- Request ID preserved / generated and returned in X-Request-ID header.
- Every request (success or safe refusal) is written to the agent audit log.
- Rate limiting via Redis (fails open when Redis is unavailable).
- Optional session_id and machine_id for correlation.
- Structured logging for every request.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import AgentQuery, AgentResponse
from backend.core.middleware import get_request_id
from backend.core.rate_limiter import rate_limit_agent
from backend.core.security import require_api_key
from backend.services.audit_service import write_audit_log

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Orchestrator dependency (singleton per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _get_db_for_audit():  # type: ignore[return]
    """Optional DB session for agent audit logging.

    Yields None (rather than raising) when the database is unavailable.
    This ensures audit-log failures never interrupt the agent response.
    """
    from backend.core.database import get_db  # noqa: PLC0415

    try:
        yield from get_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AgentAPI] DB unavailable for audit log: %s", type(exc).__name__)
        yield None


@router.post(
    "/agent/query",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Agentic maintenance query",
    description=(
        "Submit a natural-language maintenance question and optional machine sensor "
        "readings.  The agent selects and executes the appropriate tools "
        "(failure prediction, anomaly detection, risk assessment, SHAP explanation, "
        "RAG knowledge retrieval) and returns a structured, evidence-grounded response.\n\n"
        "**Safety**  \n"
        "This is a decision-support prototype.  All consequential maintenance "
        "decisions must be reviewed by qualified personnel.  The agent will refuse "
        "requests for machine control, code execution, or knowledge-base modification.\n\n"
        "Requires a valid X-API-Key header."
    ),
    tags=["Agent"],
    dependencies=[Depends(require_api_key), Depends(rate_limit_agent)],
)
def agent_query(
    body: AgentQuery,
    request: Request,
    response: Response,
    db: Session | None = Depends(_get_db_for_audit),
) -> AgentResponse:
    """Run the agentic AI pipeline for one maintenance query."""
    request_id = get_request_id(request)
    response.headers["X-Request-ID"] = request_id

    start_time = time.perf_counter()

    try:
        orchestrator = _get_orchestrator()
        agent_response: AgentResponse = orchestrator.run(body)
    except Exception as exc:
        # Catch any unexpected internal errors — never leak tracebacks externally
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception("[AgentAPI] Unexpected error request_id=%s: %s", request_id, exc)
        # Attempt audit log for unexpected failure (best-effort)
        _safe_audit(
            db=db,
            request_id=request_id,
            body=body,
            status="error",
            tools_used=[],
            summary="Unexpected internal error.",
            latency_ms=elapsed_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Structured logging ───────────────────────────────────────────────────
    logger.info(
        "[AgentAPI] request_id=%s status=%s tools=%s latency_ms=%d",
        request_id,
        agent_response.status,
        agent_response.tools_used,
        elapsed_ms,
    )

    # ── Audit log (best-effort — never blocks response) ──────────────────────
    _safe_audit(
        db=db,
        request_id=request_id,
        body=body,
        status=agent_response.status,
        tools_used=agent_response.tools_used,
        summary=_build_summary(agent_response),
        latency_ms=elapsed_ms,
    )

    return agent_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summary(resp: AgentResponse) -> str:
    """Build a concise, non-sensitive summary of the agent response."""
    # First 200 chars of the answer — no chain-of-thought, no secrets
    answer_excerpt = (resp.answer or "")[:200]
    return f"status={resp.status} tools={resp.tools_used} answer_excerpt={answer_excerpt!r}"


def _safe_audit(
    *,
    db: Session,
    request_id: str,
    body: AgentQuery,
    status: str,
    tools_used: list[str],
    summary: str,
    latency_ms: int,
) -> None:
    """Write to audit log — never raises, never blocks the response."""
    try:
        write_audit_log(
            db,
            request_id=request_id,
            query=body.query,
            status=status,
            session_id=body.session_id,
            machine_id=body.machine_id,
            tools_used=tools_used,
            response_summary=summary,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AgentAPI] Audit log write failed: %s", exc)
