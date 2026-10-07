"""
backend/api/whatif.py
======================
Phase 11: What-If Analysis API

Endpoint
--------
POST /api/v1/whatif/analyze

Accepts a baseline + scenario set of the six ML features and returns a
structured comparison of model-estimated outcomes.

DISCLAIMER
----------
All results are model simulations based on the AI4I 2020 synthetic training
dataset. They are NOT guaranteed physical outcomes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from backend.core.middleware import get_request_id
from backend.core.security import require_api_key
from backend.services.whatif_service import (
    MachineInputs,
    WhatIfResult,
    run_whatif,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["What-If Analysis"])

# ── Request / Response schemas ────────────────────────────────────────────────

_TYPE_DESCRIPTION = "Product quality variant: 'L', 'M', or 'H'."


class WhatIfInputs(BaseModel):
    """Six ML feature inputs for one what-if scenario."""

    air_temperature_k: float = Field(
        ...,
        alias="Air temperature [K]",
        description="Air temperature in Kelvin (~295–305 K).",
        examples=[300.0],
    )
    process_temperature_k: float = Field(
        ...,
        alias="Process temperature [K]",
        description="Process temperature in Kelvin (~305–315 K).",
        examples=[310.0],
    )
    rotational_speed_rpm: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        description="Rotational speed in RPM (~1000–3000 rpm).",
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ...,
        alias="Torque [Nm]",
        description="Torque in Newton-metres (~0–80 Nm).",
        examples=[45.0],
    )
    tool_wear_min: float = Field(
        ...,
        alias="Tool wear [min]",
        description="Cumulative tool wear in minutes (0–300 min).",
        examples=[100.0],
    )
    type_: str = Field(
        ...,
        alias="Type",
        description=_TYPE_DESCRIPTION,
        examples=["M"],
    )

    model_config = {"populate_by_name": True}

    @field_validator("type_", mode="before")
    @classmethod
    def validate_type(cls, v: object) -> str:
        if not isinstance(v, str) or v.upper() not in {"L", "M", "H"}:
            raise ValueError("Type must be one of 'L', 'M', 'H'.")
        return v.upper()


class WhatIfRequest(BaseModel):
    """Request body for POST /api/v1/whatif/analyze."""

    baseline: WhatIfInputs
    scenario: WhatIfInputs


class AssessmentSnapshotResponse(BaseModel):
    failure_probability: float
    anomaly_score: float
    risk_score: float
    risk_level: str


class WhatIfChangeResponse(BaseModel):
    failure_probability_delta: float
    anomaly_score_delta: float
    risk_score_delta: float
    risk_level_changed: bool


class WhatIfResponse(BaseModel):
    """Response body for POST /api/v1/whatif/analyze."""

    baseline: AssessmentSnapshotResponse
    scenario: AssessmentSnapshotResponse
    change: WhatIfChangeResponse
    out_of_distribution_warning: str | None
    disclaimer: str = (
        "What-if results are model simulations based on the AI4I 2020 synthetic "
        "training dataset. They are model-estimated changes, not guaranteed outcomes. "
        "Results outside the training distribution should be interpreted cautiously."
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/whatif/analyze",
    response_model=WhatIfResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a what-if analysis",
    description=(
        "Compare ML-estimated outcomes between baseline and modified machine "
        "operating conditions. All results are model simulations based on the "
        "AI4I 2020 synthetic dataset — not guaranteed physical outcomes. "
        "Requires a valid X-API-Key header."
    ),
    responses={
        403: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error or invalid inputs"},
        503: {"description": "ML artifacts unavailable"},
    },
    dependencies=[Depends(require_api_key)],
)
def whatif_analyze(
    body: WhatIfRequest,
    request: Request,
) -> WhatIfResponse:
    request_id = get_request_id(request)

    baseline = MachineInputs(
        air_temp_k=body.baseline.air_temperature_k,
        process_temp_k=body.baseline.process_temperature_k,
        rotational_speed_rpm=body.baseline.rotational_speed_rpm,
        torque_nm=body.baseline.torque_nm,
        tool_wear_min=body.baseline.tool_wear_min,
        type_=body.baseline.type_,
    )
    scenario = MachineInputs(
        air_temp_k=body.scenario.air_temperature_k,
        process_temp_k=body.scenario.process_temperature_k,
        rotational_speed_rpm=body.scenario.rotational_speed_rpm,
        torque_nm=body.scenario.torque_nm,
        tool_wear_min=body.scenario.tool_wear_min,
        type_=body.scenario.type_,
    )

    try:
        result: WhatIfResult = run_whatif(baseline, scenario)
    except FileNotFoundError as exc:
        logger.error(
            "[WhatIf] ML artifacts missing request_id=%s: %s", request_id, exc
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
            "[WhatIf] ML pipeline error request_id=%s: %s", request_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The what-if analysis pipeline encountered an error.",
        ) from exc

    logger.info(
        "[WhatIf] Completed request_id=%s risk_level_changed=%s ood_warning=%s",
        request_id,
        result.risk_level_changed,
        bool(result.out_of_distribution_warning),
    )

    return WhatIfResponse(
        baseline=AssessmentSnapshotResponse(
            failure_probability=result.baseline.failure_probability,
            anomaly_score=result.baseline.anomaly_score,
            risk_score=result.baseline.risk_score,
            risk_level=result.baseline.risk_level,
        ),
        scenario=AssessmentSnapshotResponse(
            failure_probability=result.scenario.failure_probability,
            anomaly_score=result.scenario.anomaly_score,
            risk_score=result.scenario.risk_score,
            risk_level=result.scenario.risk_level,
        ),
        change=WhatIfChangeResponse(
            failure_probability_delta=result.failure_probability_delta,
            anomaly_score_delta=result.anomaly_score_delta,
            risk_score_delta=result.risk_score_delta,
            risk_level_changed=result.risk_level_changed,
        ),
        out_of_distribution_warning=result.out_of_distribution_warning,
    )
