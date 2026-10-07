"""
backend/api/assessments.py
===========================
Phase 8: Machine assessment endpoints.

Endpoints
---------
POST /api/v1/machines/{machine_id}/assess         — run + persist assessment
GET  /api/v1/machines/{machine_id}/assessments    — list assessment history

DESIGN
------
- The POST endpoint calls ml.assessment.assess_machine() via the service layer.
- ML artifacts must be present — FileNotFoundError → 503.
- Bad machine_id → 404.
- Rate limiting protects POST via Redis (fails open when Redis is unavailable).
- All errors return clean messages; stack traces are never exposed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.middleware import get_request_id
from backend.core.rate_limiter import rate_limit_assess
from backend.core.security import require_api_key
from backend.schemas.machine import AssessmentRequest, AssessmentResponse
from backend.services.assessment_service import list_assessments, run_and_persist_assessment
from backend.services.machine_service import get_machine_by_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Assessments"])


@router.post(
    "/machines/{machine_id}/assess",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run and persist a machine assessment",
    description=(
        "Submit sensor readings for a registered machine. "
        "The assessment pipeline (failure prediction, anomaly detection, risk scoring) "
        "is executed and the result is persisted. "
        "This is a decision-support tool — all results require human review. "
        "Requires a valid X-API-Key header."
    ),
    responses={
        403: {"description": "Invalid or missing API key"},
        404: {"description": "Machine not found"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "ML artifacts unavailable"},
    },
    dependencies=[Depends(require_api_key), Depends(rate_limit_assess)],
)
def assess_machine_endpoint(
    machine_id: int,
    body: AssessmentRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    request_id = get_request_id(request)

    machine = get_machine_by_id(db, machine_id)
    if machine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine with id={machine_id} not found.",
        )

    try:
        assessment = run_and_persist_assessment(db, machine_id=machine_id, req=body)
        logger.info(
            "[Assessments] Completed assessment_id=%d machine_id=%d risk=%s request_id=%s",
            assessment.id,
            machine_id,
            assessment.risk_level,
            request_id,
        )
        return assessment  # type: ignore[return-value]

    except FileNotFoundError as exc:
        logger.error(
            "[Assessments] ML artifact missing machine_id=%d request_id=%s: %s",
            machine_id,
            request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model artifacts are unavailable. Please contact the administrator.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.error(
            "[Assessments] ML pipeline error machine_id=%d request_id=%s: %s",
            machine_id,
            request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The assessment pipeline encountered an error.",
        ) from exc

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[Assessments] Unexpected error machine_id=%d request_id=%s: %s",
            machine_id,
            request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        ) from exc


@router.get(
    "/machines/{machine_id}/assessments",
    response_model=list[AssessmentResponse],
    summary="List assessment history for a machine",
    description="Return paginated assessment history for a registered machine (newest first).",
    responses={404: {"description": "Machine not found"}},
)
def list_assessments_endpoint(
    machine_id: int,
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[AssessmentResponse]:
    request_id = get_request_id(request)

    machine = get_machine_by_id(db, machine_id)
    if machine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine with id={machine_id} not found.",
        )

    assessments = list_assessments(db, machine_id=machine_id, skip=skip, limit=limit)
    logger.info(
        "[Assessments] list machine_id=%d count=%d request_id=%s",
        machine_id,
        len(assessments),
        request_id,
    )
    return assessments  # type: ignore[return-value]
