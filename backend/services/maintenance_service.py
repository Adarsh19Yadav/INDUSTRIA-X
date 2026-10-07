"""
backend/services/maintenance_service.py
=========================================
Phase 11: Maintenance Prioritization Service

Computes a deterministic priority score for each machine based on its
latest assessment results and returns a ranked list.

PRIORITY FORMULA
----------------
    priority_score = W_RISK × risk_score
                   + W_FAILURE × failure_probability
                   + W_ANOMALY × anomaly_score

Default weights (configurable module-level constants):
    W_RISK    = 0.50
    W_FAILURE = 0.30
    W_ANOMALY = 0.20

IMPORTANT DISCLAIMER
--------------------
These are PROTOTYPE prioritization weights for a decision-support system.
They are NOT an industrial safety standard. Weight values should be reviewed
and calibrated by qualified maintenance engineers before operational use.

RISK LEVEL ORDERING
-------------------
CRITICAL machines rank ahead of WARNING, WARNING ahead of NORMAL, regardless
of raw priority_score ties — the risk_level acts as the primary sort key.

LANGUAGE
--------
Results use "Recommended for maintenance review" — NOT "Machine will fail".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Configurable priority weights ─────────────────────────────────────────────
# These are PROTOTYPE weights — not an industrial safety standard.
# Modify these constants to recalibrate the prioritization formula.

#: Weight applied to the risk_score component (from ML risk engine).
W_RISK: float = 0.50

#: Weight applied to the failure_probability component (from ML failure model).
W_FAILURE: float = 0.30

#: Weight applied to the anomaly_score component (from anomaly detector).
W_ANOMALY: float = 0.20

# Sanity check: weights must sum to 1.0
assert abs(W_RISK + W_FAILURE + W_ANOMALY - 1.0) < 1e-9, (
    "Maintenance priority weights must sum to 1.0"
)

# Risk level numeric ordering for stable sorting (CRITICAL = highest priority)
_RISK_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "WARNING":  1,
    "NORMAL":   2,
}
_DEFAULT_RISK_ORDER: int = 3  # unassessed machines rank last


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MachinePriorityInput:
    """Input record for a single machine."""

    machine_id: int | str
    risk_level: str
    risk_score: float
    failure_probability: float
    anomaly_score: float


@dataclass(frozen=True)
class MachinePriorityResult:
    """Prioritization result for a single machine."""

    machine_id: int | str
    risk_level: str
    risk_score: float
    failure_probability: float
    anomaly_score: float
    priority_score: float
    priority_rank: int
    recommended_review: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "failure_probability": self.failure_probability,
            "anomaly_score": self.anomaly_score,
            "priority_score": self.priority_score,
            "priority_rank": self.priority_rank,
            "recommended_review": self.recommended_review,
        }


# ── Recommendation labels ─────────────────────────────────────────────────────

def _recommended_review_label(risk_level: str) -> str:
    """Return the appropriate recommendation string for a risk level."""
    if risk_level == "CRITICAL":
        return "Recommended for maintenance review — elevated risk indicators detected"
    if risk_level == "WARNING":
        return "Recommended for maintenance review — monitor closely"
    return "Continue routine monitoring"


# ── Core service function ─────────────────────────────────────────────────────


def compute_priority_score(
    risk_score: float,
    failure_probability: float,
    anomaly_score: float,
    w_risk: float = W_RISK,
    w_failure: float = W_FAILURE,
    w_anomaly: float = W_ANOMALY,
) -> float:
    """Compute a deterministic priority score for one machine.

    Formula:
        priority_score = w_risk × risk_score
                       + w_failure × failure_probability
                       + w_anomaly × anomaly_score

    Parameters
    ----------
    risk_score:
        Risk score from the ML risk engine. Must be in [0, 1].
    failure_probability:
        Failure probability from the ML failure model. Must be in [0, 1].
    anomaly_score:
        Normalised anomaly score from the anomaly detector. Must be in [0, 1].
    w_risk, w_failure, w_anomaly:
        Weights for each component. Must sum to 1.0.

    Returns
    -------
    float
        Priority score in [0, 1], rounded to 6 decimal places.

    Raises
    ------
    ValueError
        If any input is outside [0, 1] or weights do not sum to 1.0.
    """
    for name, val in [
        ("risk_score", risk_score),
        ("failure_probability", failure_probability),
        ("anomaly_score", anomaly_score),
    ]:
        if not isinstance(val, (int, float)) or val is None:
            raise ValueError(f"'{name}' must be a number.")
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"'{name}' must be in [0, 1], got {val}.")

    if abs(w_risk + w_failure + w_anomaly - 1.0) > 1e-9:
        raise ValueError(
            f"Priority weights must sum to 1.0, got "
            f"w_risk={w_risk} + w_failure={w_failure} + w_anomaly={w_anomaly} "
            f"= {w_risk + w_failure + w_anomaly:.6f}."
        )

    raw = w_risk * risk_score + w_failure * failure_probability + w_anomaly * anomaly_score
    return round(float(np.clip(raw, 0.0, 1.0)), 6)


def rank_machines(machines: list[MachinePriorityInput]) -> list[MachinePriorityResult]:
    """Rank a list of machines by maintenance priority.

    Sort order (stable, deterministic):
      1. risk_level order: CRITICAL → WARNING → NORMAL → unrecognised
      2. priority_score descending (higher = more urgent)

    Parameters
    ----------
    machines:
        List of machine priority inputs with assessment data.

    Returns
    -------
    list[MachinePriorityResult]
        Ranked list (rank 1 = highest priority), with recommendation labels.

    Notes
    -----
    All scores in the input must be in [0, 1].
    If a list is empty, an empty list is returned.
    """
    if not machines:
        return []

    # Compute priority scores
    scored: list[tuple[MachinePriorityInput, float]] = []
    for m in machines:
        score = compute_priority_score(
            risk_score=m.risk_score,
            failure_probability=m.failure_probability,
            anomaly_score=m.anomaly_score,
        )
        scored.append((m, score))

    # Sort: primary = risk_level order (ascending = higher priority first),
    #        secondary = priority_score descending
    scored.sort(
        key=lambda pair: (
            _RISK_ORDER.get(pair[0].risk_level, _DEFAULT_RISK_ORDER),
            -pair[1],
        )
    )

    results: list[MachinePriorityResult] = []
    for rank, (m, score) in enumerate(scored, start=1):
        results.append(
            MachinePriorityResult(
                machine_id=m.machine_id,
                risk_level=m.risk_level,
                risk_score=m.risk_score,
                failure_probability=m.failure_probability,
                anomaly_score=m.anomaly_score,
                priority_score=score,
                priority_rank=rank,
                recommended_review=_recommended_review_label(m.risk_level),
            )
        )

    return results
