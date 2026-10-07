"""
backend/schemas/machine.py
==========================
Pydantic request/response schemas for Machine and Assessment endpoints.

Phase 8: all API I/O is validated here — ORM models are never returned directly.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

# ── Validation helpers ────────────────────────────────────────────────────────

# Only allow alphanumeric, dash, underscore, dot — prevents log injection
# and path traversal in machine identifiers.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")


def _validate_machine_identifier(v: str) -> str:
    """Reject machine identifiers with unsafe characters."""
    v = v.strip()
    if not _SAFE_ID_RE.match(v):
        raise ValueError(
            "machine_identifier must be 1–128 characters containing only "
            "letters, digits, hyphens, underscores, or dots."
        )
    return v


# ── Machine schemas ───────────────────────────────────────────────────────────


class MachineCreate(BaseModel):
    """Request body for POST /api/v1/machines."""

    machine_identifier: Annotated[str, Field(min_length=1, max_length=128)]
    type: str | None = Field(
        default=None,
        max_length=64,
        description="Optional machine type/category label.",
        examples=["CNC_LATHE"],
    )

    @field_validator("machine_identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        return _validate_machine_identifier(v)


class MachineResponse(BaseModel):
    """Response body for machine endpoints."""

    id: int
    machine_identifier: str
    type: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Assessment schemas ────────────────────────────────────────────────────────


class AssessmentRequest(BaseModel):
    """Request body for POST /api/v1/machines/{machine_id}/assess.

    Contains the six features required by the ML pipeline.
    """

    air_temperature_k: float = Field(
        ...,
        description="Air temperature in Kelvin (~295–305 K).",
        examples=[300.1],
    )
    process_temperature_k: float = Field(
        ...,
        description="Process temperature in Kelvin (~305–315 K).",
        examples=[310.2],
    )
    rotational_speed_rpm: float = Field(
        ...,
        description="Rotational speed in RPM (~1100–2900 rpm).",
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ...,
        description="Torque in Newton-metres (~3–77 Nm).",
        examples=[45.0],
    )
    tool_wear_min: float = Field(
        ...,
        description="Cumulative tool wear in minutes (0–254 min).",
        examples=[120.0],
    )
    type_: str = Field(
        ...,
        alias="type",
        description="Product quality variant: 'L', 'M', or 'H'.",
        examples=["M"],
    )

    model_config = {"populate_by_name": True}

    @field_validator("type_", mode="before")
    @classmethod
    def validate_type(cls, v: object) -> str:
        if not isinstance(v, str) or v.upper() not in {"L", "M", "H"}:
            raise ValueError("type must be one of 'L', 'M', 'H'.")
        return v.upper()


class AssessmentResponse(BaseModel):
    """Response body for assessment endpoints."""

    id: int
    machine_id: int
    failure_probability: float
    anomaly_score: float
    risk_score: float
    risk_level: str
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}
