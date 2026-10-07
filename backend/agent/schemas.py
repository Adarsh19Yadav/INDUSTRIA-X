"""backend/agent/schemas.py — Pydantic models for the Phase 7 agent API.

All request/response bodies for the agent endpoint use these models.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Machine Input (the six approved model features)
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"L", "M", "H"})


class MachineInput(BaseModel):
    """Six-feature machine observation accepted by all Phase 3–5 ML tools."""

    air_temperature_k: float = Field(
        ...,
        alias="Air temperature [K]",
        description="Air temperature in Kelvin (expected ~295–305 K).",
        examples=[300.1],
    )
    process_temperature_k: float = Field(
        ...,
        alias="Process temperature [K]",
        description="Process temperature in Kelvin (expected ~305–315 K).",
        examples=[310.2],
    )
    rotational_speed_rpm: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        description="Rotational speed in RPM (expected ~1100–2900 rpm).",
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ...,
        alias="Torque [Nm]",
        description="Torque in Newton-metres (expected ~3–77 Nm).",
        examples=[45.0],
    )
    tool_wear_min: float = Field(
        ...,
        alias="Tool wear [min]",
        description="Cumulative tool wear in minutes (0–254 min).",
        examples=[120.0],
    )
    type_: str = Field(
        ...,
        alias="Type",
        description="Product quality variant: 'L', 'M', or 'H'.",
        examples=["M"],
    )

    model_config = {"populate_by_name": True}

    @field_validator("type_", mode="before")
    @classmethod
    def validate_type(cls, v: Any) -> str:
        if not isinstance(v, str) or v.upper() not in _VALID_TYPES:
            raise ValueError(
                f"'Type' must be one of {sorted(_VALID_TYPES)}, got '{v}'."
            )
        return v.upper()

    @field_validator(
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        mode="before",
    )
    @classmethod
    def validate_finite_number(cls, v: Any) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Expected a finite number, got {v!r}.")
        if math.isnan(fv) or math.isinf(fv):
            raise ValueError(f"Value must be finite, got {fv}.")
        return fv


# ---------------------------------------------------------------------------
# Agent request
# ---------------------------------------------------------------------------


class AgentQuery(BaseModel):
    """Request body for POST /api/v1/agent/query."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language maintenance question.",
        examples=["Why is this machine high risk and what maintenance should be considered?"],
    )
    machine_input: MachineInput | None = Field(
        default=None,
        description=(
            "Six-feature machine observation. Required for tools that perform ML inference. "
            "May be omitted for knowledge-base-only queries."
        ),
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional session identifier for grouping multi-turn interactions. "
            "When provided, the audit log records this session ID for correlation. "
            "The current stateless behaviour is unchanged when omitted."
        ),
        examples=["sess-abc123"],
    )
    machine_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional machine business identifier (e.g. 'MACHINE-001'). "
            "Recorded in the audit log for traceability."
        ),
        examples=["MACHINE-001"],
    )


# ---------------------------------------------------------------------------
# Evidence items
# ---------------------------------------------------------------------------


class ToolEvidence(BaseModel):
    """Structured evidence returned by a single tool call."""

    tool: str = Field(description="Name of the tool that produced this evidence.")
    result: dict[str, Any] = Field(description="Tool output as a dictionary.")


class RAGSource(BaseModel):
    """A single retrieved maintenance knowledge chunk."""

    source: str
    chunk_index: int
    distance: float
    excerpt: str = Field(description="First 300 characters of the retrieved chunk.")


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Response body for POST /api/v1/agent/query."""

    answer: str = Field(description="Concise, evidence-grounded decision-support response.")
    tools_used: list[str] = Field(description="Ordered list of tools the agent invoked.")
    evidence: list[ToolEvidence] = Field(
        default_factory=list,
        description="Structured evidence from each tool call.",
    )
    sources: list[RAGSource] = Field(
        default_factory=list,
        description="Retrieved maintenance knowledge-base sources.",
    )
    status: Literal["success", "error", "partial"] = Field(
        default="success",
        description="Overall execution status.",
    )
    error_detail: str | None = Field(
        default=None,
        description="Set when status is 'error' or 'partial'.",
    )
