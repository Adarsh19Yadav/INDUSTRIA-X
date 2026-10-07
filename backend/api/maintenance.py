"""
backend/api/maintenance.py
============================
Phase 11: Maintenance Prioritization API

Endpoint
--------
GET /api/v1/maintenance/priority

Returns a ranked list of all machines with assessments, ordered by
deterministic priority score.

DISCLAIMER
----------
Priority scores are prototype decision-support outputs. They are NOT an
industrial safety standard. All recommendations require human review.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.services.machine_service import list_machines
from backend.services.assessment_service import list_assessments
from backend.services.maintenance_service import (
    MachinePriorityInput,
    MachinePriorityResult,
    rank_machines,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Maintenance"])

# ── Response model (inline Pydantic) ─────────────────────────────────────────

from pydantic import BaseModel


class MachinePriorityResponse(BaseModel):
    machine_id: int
    machine_identifier: str
    machine_type: str | None
    risk_level: str
    risk_score: float
    failure_probability: float
    anomaly_score: float
    priority_score: float
    priority_rank: int
    recommended_review: str
    disclaimer: str = (
        "Priority scores are prototype decision-support outputs based on the "
        "AI4I 2020 synthetic dataset. They are NOT an industrial safety standard."
    )


class MaintenancePriorityResponse(BaseModel):
    machines: list[MachinePriorityResponse]
    total: int
    disclaimer: str = (
        "Maintenance priority rankings are model-estimated and require qualified "
        "human review before any action is taken. Results are based on the "
        "AI4I 2020 synthetic training dataset."
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get(
    "/maintenance/priority",
    response_model=MaintenancePriorityResponse,
    summary="Get maintenance priority ranking for all machines",
    description=(
        "Returns all machines that have at least one assessment, ranked by a "
        "deterministic priority score. CRITICAL machines rank first. "
        "Priority scores are prototype decision-support outputs — not an "
        "industrial safety standard."
    ),
)
def get_maintenance_priority(
    db: Session = Depends(get_db),
) -> MaintenancePriorityResponse:
    # Fetch all machines (up to 1000 — sufficient for the prototype)
    machines = list_machines(db, skip=0, limit=1000)

    if not machines:
        return MaintenancePriorityResponse(machines=[], total=0)

    # Fetch the latest assessment for each machine
    priority_inputs: list[tuple[MachinePriorityInput, object]] = []  # (input, machine_orm)
    for machine in machines:
        assessments = list_assessments(db, machine_id=machine.id, skip=0, limit=1)
        if not assessments:
            # Skip machines with no assessments — cannot prioritise without data
            continue
        latest = assessments[0]
        priority_inputs.append((
            MachinePriorityInput(
                machine_id=machine.id,
                risk_level=latest.risk_level,
                risk_score=latest.risk_score,
                failure_probability=latest.failure_probability,
                anomaly_score=latest.anomaly_score,
            ),
            machine,
        ))

    if not priority_inputs:
        return MaintenancePriorityResponse(machines=[], total=0)

    # Build a mapping from machine_id to ORM object for metadata lookup
    machine_map = {m.id: m for m in machines}

    # Extract just the priority inputs for ranking
    inputs = [pi for pi, _ in priority_inputs]
    ranked: list[MachinePriorityResult] = rank_machines(inputs)

    response_machines: list[MachinePriorityResponse] = []
    for result in ranked:
        machine = machine_map[result.machine_id]
        response_machines.append(
            MachinePriorityResponse(
                machine_id=int(result.machine_id),
                machine_identifier=machine.machine_identifier,
                machine_type=machine.type,
                risk_level=result.risk_level,
                risk_score=result.risk_score,
                failure_probability=result.failure_probability,
                anomaly_score=result.anomaly_score,
                priority_score=result.priority_score,
                priority_rank=result.priority_rank,
                recommended_review=result.recommended_review,
            )
        )

    logger.info(
        "[Maintenance] Priority ranking computed for %d machine(s)", len(response_machines)
    )

    return MaintenancePriorityResponse(
        machines=response_machines,
        total=len(response_machines),
    )
