"""
backend/api/machines.py
========================
Phase 8: Machine identity CRUD endpoints.

Endpoints
---------
GET  /api/v1/machines               — list all machines (paginated)
GET  /api/v1/machines/{machine_id}  — retrieve one machine by integer ID
POST /api/v1/machines               — register a new machine

Assessment sub-resource lives in backend/api/assessments.py.

SECURITY
--------
- All inputs are validated with Pydantic (MachineCreate).
- machine_identifier is restricted to safe characters (no path traversal,
  no log injection).
- No raw SQL — all queries go through SQLAlchemy ORM.
- Stack traces are never returned to the caller.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.middleware import get_request_id
from backend.core.security import require_api_key
from backend.schemas.machine import MachineCreate, MachineResponse
from backend.services.machine_service import (
    create_machine,
    get_machine_by_id,
    list_machines,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/machines", tags=["Machines"])


@router.get(
    "",
    response_model=list[MachineResponse],
    summary="List machines",
    description="Return a paginated list of registered machines.",
)
def list_machines_endpoint(
    request: Request,
    skip: int = Query(default=0, ge=0, description="Number of records to skip."),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum records to return."),
    db: Session = Depends(get_db),
) -> list[MachineResponse]:
    request_id = get_request_id(request)
    logger.info("[Machines] list request_id=%s skip=%d limit=%d", request_id, skip, limit)
    machines = list_machines(db, skip=skip, limit=limit)
    return machines  # type: ignore[return-value]


@router.get(
    "/{machine_id}",
    response_model=MachineResponse,
    summary="Get machine by ID",
    description="Return a single machine by its integer database ID.",
    responses={404: {"description": "Machine not found"}},
)
def get_machine_endpoint(
    machine_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> MachineResponse:
    request_id = get_request_id(request)
    machine = get_machine_by_id(db, machine_id)
    if machine is None:
        logger.info(
            "[Machines] Not found machine_id=%d request_id=%s", machine_id, request_id
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine with id={machine_id} not found.",
        )
    return machine  # type: ignore[return-value]


@router.post(
    "",
    response_model=MachineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a machine",
    description=(
        "Register a new industrial machine. "
        "The machine_identifier must be unique across the system. "
        "Requires a valid X-API-Key header."
    ),
    responses={
        403: {"description": "Invalid or missing API key"},
        409: {"description": "Machine with that identifier already exists"},
        422: {"description": "Validation error"},
    },
    dependencies=[Depends(require_api_key)],
)
def create_machine_endpoint(
    body: MachineCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> MachineResponse:
    request_id = get_request_id(request)
    try:
        machine = create_machine(db, body)
        logger.info(
            "[Machines] Created id=%d ident=%r request_id=%s",
            machine.id,
            machine.machine_identifier,
            request_id,
        )
        return machine  # type: ignore[return-value]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Machines] Unexpected error request_id=%s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        ) from exc
