"""
backend/api/sustainability.py
=============================
Phase 10: Sustainability & Energy Intelligence API

Endpoints
---------
POST /api/v1/sustainability/estimate
    Compute estimated power, energy, and CO₂e from machine operating parameters.

GET  /api/v1/sustainability/insights
    Return deterministic textual insights about energy and carbon levers.

All returned values are clearly labelled as ESTIMATES.
The AI4I dataset contains no real electrical energy measurements.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from backend.core.security import require_api_key

from backend.services.sustainability_service import (
    SustainabilityEstimate,
    compute_sustainability_estimate,
    DEFAULT_EFFICIENCY,
    DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sustainability", tags=["Sustainability"])


# ── Request schema ─────────────────────────────────────────────────────────────


class SustainabilityEstimateRequest(BaseModel):
    """
    Request body for POST /api/v1/sustainability/estimate.

    Uses the same sensor fields available in the AI4I dataset where possible.
    All fields must be physically plausible positive values.
    """

    machine_type: str | None = Field(
        default=None,
        description="Optional machine quality variant: 'L', 'M', or 'H'.",
        examples=["M"],
    )
    rotational_speed_rpm: float = Field(
        ...,
        gt=0,
        le=10_000,
        description="Rotational speed in RPM (must be > 0).",
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ...,
        gt=0,
        le=1_000,
        description="Torque in Newton-metres (must be > 0).",
        examples=[45.0],
    )
    operating_hours: float = Field(
        default=8.0,
        gt=0,
        le=8_760,  # max one year
        description="Operating duration in hours (default 8 h).",
        examples=[8.0],
    )

    # Optional configuration overrides
    efficiency: float = Field(
        default=DEFAULT_EFFICIENCY,
        gt=0,
        le=1.0,
        description=(
            "Assumed mechanical-to-electrical drivetrain efficiency (0–1). "
            f"Default is {DEFAULT_EFFICIENCY} (illustrative)."
        ),
        examples=[DEFAULT_EFFICIENCY],
    )
    emission_factor_kg_co2e_per_kwh: float = Field(
        default=DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH,
        ge=0,
        description=(
            "Grid emission factor in kg CO₂e per kWh. "
            f"Default is {DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH} (generic illustrative value). "
            "Substitute a grid-specific factor for meaningful country-level analysis."
        ),
        examples=[DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH],
    )

    @field_validator("machine_type", mode="before")
    @classmethod
    def _normalise_machine_type(cls, v: object) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("machine_type must be a string or null.")
        v_upper = v.strip().upper()
        if v_upper not in {"L", "M", "H"}:
            raise ValueError("machine_type must be 'L', 'M', or 'H'.")
        return v_upper


# ── Insights response schema ───────────────────────────────────────────────────


class SustainabilityInsightsResponse(BaseModel):
    """Deterministic textual insights about energy and sustainability levers."""

    insights: list[str]
    disclaimer: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post(
    "/estimate",
    response_model=SustainabilityEstimate,
    summary="Estimate energy and carbon footprint",
    description=(
        "Compute estimated mechanical power, electrical power, energy consumption, "
        "and CO₂e for given machine operating parameters. "
        "All values are ENGINEERING ESTIMATES derived from kinematic inputs. "
        "They are NOT measured electrical consumption figures. "
        "Requires a valid X-API-Key header."
    ),
    responses={
        403: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error — check input ranges."},
    },
    dependencies=[Depends(require_api_key)],
)
def estimate_sustainability(body: SustainabilityEstimateRequest) -> SustainabilityEstimate:
    """Return a deterministic sustainability estimate for the supplied parameters."""
    logger.info(
        "[Sustainability] estimate rpm=%.1f torque=%.2f hours=%.2f efficiency=%.3f ef=%.4f",
        body.rotational_speed_rpm,
        body.torque_nm,
        body.operating_hours,
        body.efficiency,
        body.emission_factor_kg_co2e_per_kwh,
    )
    return compute_sustainability_estimate(
        rotational_speed_rpm=body.rotational_speed_rpm,
        torque_nm=body.torque_nm,
        operating_hours=body.operating_hours,
        machine_type=body.machine_type,
        efficiency=body.efficiency,
        emission_factor_kg_co2e_per_kwh=body.emission_factor_kg_co2e_per_kwh,
    )


@router.get(
    "/insights",
    response_model=SustainabilityInsightsResponse,
    summary="Get sustainability insights",
    description=(
        "Return deterministic textual insights about energy and carbon levers "
        "for industrial machine operation. Insights are based on engineering principles, "
        "not LLM inference or fabricated statistics."
    ),
)
def get_sustainability_insights() -> SustainabilityInsightsResponse:
    """Return deterministic engineering insights about sustainability levers."""
    insights = [
        "Higher torque and rotational speed increase estimated mechanical power (P = τ × ω).",
        "Reducing unnecessary operating hours directly reduces estimated energy consumption.",
        "Improving drivetrain efficiency (motor, gearbox, coupling) reduces estimated electrical input power for the same mechanical output.",
        "Estimated CO₂e is proportional to estimated energy and the grid emission factor.",
        "Switching to a lower-carbon electricity source would reduce estimated CO₂e even if operating parameters remain unchanged.",
        "Actual energy optimisation requires measured electrical consumption data from calibrated energy meters.",
        "Predictive maintenance can reduce unplanned downtime and associated energy waste from idle or degraded operation.",
    ]
    disclaimer = (
        "These insights are based on general engineering principles applied to the "
        "AI4I 2020 synthetic dataset. They do not represent measured data from any "
        "real industrial facility. No specific energy savings or carbon reductions are claimed."
    )
    return SustainabilityInsightsResponse(insights=insights, disclaimer=disclaimer)
