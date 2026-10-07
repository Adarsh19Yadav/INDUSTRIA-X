"""
INDUSTRIA-X Backend
===================
FastAPI application entry point.

Phase 1:  application shell with health check endpoints.
Phase 6:  RAG router.
Phase 7:  Agent router.
Phase 8:  Machines/Assessments routers, RequestID middleware, structured logging.
Phase 10: Sustainability & Energy Intelligence router.
Phase 11: What-If Analysis & Maintenance Prioritization routers.
Phase 12: Simulated Real-Time Monitoring router.
"""

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.middleware import RequestIDMiddleware
from backend.api.health import router as health_router
from backend.api.rag import router as rag_router
from backend.api.agent import router as agent_router
from backend.api.machines import router as machines_router
from backend.api.assessments import router as assessments_router
from backend.api.sustainability import router as sustainability_router
from backend.api.whatif import router as whatif_router
from backend.api.maintenance import router as maintenance_router
from backend.api.simulation import router as simulation_router


# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: initialise DB pool, Redis connection, load ML artifacts (Phase 2+).
    Shutdown: close connections gracefully.
    """
    # ── Phase 1: basic startup ──────────────────────────────────────────
    logger.info("[INDUSTRIA-X] Starting in '%s' mode", settings.APP_ENV)

    # ── Phase 6: RAG — ingest knowledge base at startup ─────────────────────
    try:
        from rag.service import get_rag_service  # noqa: PLC0415

        rag_svc = get_rag_service()
        count = rag_svc.ingest()
        logger.info("[INDUSTRIA-X] RAG knowledge base ingested — %d chunk(s) indexed.", count)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[INDUSTRIA-X] RAG ingest failed (non-fatal): %s", exc)

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("[INDUSTRIA-X] Shutting down")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI application."""
    app = FastAPI(
        title="INDUSTRIA-X API",
        description=(
            "Agentic AI for Predictive Maintenance and Sustainable Industrial Operations. "
            "SDG 9 — Industry, Innovation and Infrastructure."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ──────────────────────────
    # CORS must wrap everything
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Phase 8: request ID tracking
    app.add_middleware(RequestIDMiddleware)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler — never expose stack traces to API consumers."""
        logger.exception("[INDUSTRIA-X] Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])

    # Phase 6: RAG
    app.include_router(rag_router, prefix="/api/v1", tags=["RAG"])

    # Phase 7: Agent
    app.include_router(agent_router, prefix="/api/v1", tags=["Agent"])

    # Phase 8: Machines + Assessments
    app.include_router(machines_router, prefix="/api/v1")
    app.include_router(assessments_router, prefix="/api/v1")

    # Phase 10: Sustainability
    app.include_router(sustainability_router, prefix="/api/v1")

    # Phase 11: What-If Analysis + Maintenance Prioritization
    app.include_router(whatif_router, prefix="/api/v1")
    app.include_router(maintenance_router, prefix="/api/v1")

    # Phase 12: Simulation
    app.include_router(simulation_router, prefix="/api/v1")

    return app


app = create_app()
