"""
backend/api/simulation.py
==========================
Phase 12: Simulation API endpoints

Endpoints
---------
POST /api/v1/simulation/start   — start (or resume) the simulation
POST /api/v1/simulation/pause   — pause the simulation
POST /api/v1/simulation/reset   — reset to initial state
POST /api/v1/simulation/step    — advance one row and return the event
GET  /api/v1/simulation/status  — current simulation status

IMPORTANT
---------
This simulation runs on the AI4I 2020 *synthetic* dataset. All events
represent historical synthetic data — not live industrial sensor readings.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.security import require_api_key

from backend.services.simulation_service import (
    SimulationState,
    get_simulation_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Simulation"])

# ── Request schemas ───────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    speed: str = Field(
        default="normal",
        description="Simulation speed hint: 'slow', 'normal', or 'fast'.",
        examples=["normal"],
    )


# ── Shared engine accessor ────────────────────────────────────────────────────


def _engine():
    return get_simulation_engine()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/simulation/start",
    status_code=status.HTTP_200_OK,
    summary="Start the simulation",
    description=(
        "Start (or resume) the simulation engine. "
        "SIMULATION only — uses the AI4I 2020 synthetic dataset. "
        "Requires a valid X-API-Key header."
    ),
    dependencies=[Depends(require_api_key)],
)
def simulation_start(body: StartRequest) -> dict:
    engine = _engine()
    try:
        engine.start(speed=body.speed)
    except FileNotFoundError as exc:
        logger.warning("[Simulation] Dataset unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Simulation dataset unavailable. "
                "Place data/raw/ai4i2020.csv in the project root to enable simulation."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.info("[Simulation] Started. speed=%s total_steps=%d", body.speed, engine.total_steps)
    return {"status": "started", "total_steps": engine.total_steps, "speed": engine.speed}


@router.post(
    "/simulation/pause",
    status_code=status.HTTP_200_OK,
    summary="Pause the simulation",
    dependencies=[Depends(require_api_key)],
)
def simulation_pause() -> dict:
    engine = _engine()
    engine.pause()
    return {"status": engine.state.value}


@router.post(
    "/simulation/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset the simulation to the initial state",
    dependencies=[Depends(require_api_key)],
)
def simulation_reset() -> dict:
    engine = _engine()
    engine.reset()
    logger.info("[Simulation] Reset.")
    return {"status": "idle"}


@router.post(
    "/simulation/step",
    status_code=status.HTTP_200_OK,
    summary="Advance one simulation step",
    description=(
        "Process exactly one dataset row and return the simulation event. "
        "Call this endpoint at your desired polling rate from the frontend. "
        "Requires a valid X-API-Key header."
    ),
    responses={
        403: {"description": "Invalid or missing API key"},
        409: {"description": "Simulation not started or already finished"},
        503: {"description": "ML artifacts or dataset unavailable"},
    },
    dependencies=[Depends(require_api_key)],
)
def simulation_step() -> dict:
    engine = _engine()
    try:
        event = engine.step()
    except IndexError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        logger.error("[Simulation] Artifact/dataset missing during step: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model artifacts or dataset are unavailable.",
        ) from exc
    except RuntimeError as exc:
        logger.error("[Simulation] ML pipeline error during step: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation step failed due to an internal error.",
        ) from exc

    return {
        "event": event.to_dict(),
        "simulation_state": engine.state.value,
        "current_step": engine.current_step,
        "total_steps": engine.total_steps,
        "new_alerts": [a.to_dict() for a in engine.alerts[-3:]],
    }


@router.get(
    "/simulation/status",
    status_code=status.HTTP_200_OK,
    summary="Get current simulation status",
)
def simulation_status() -> dict:
    engine = _engine()
    return engine.status()
