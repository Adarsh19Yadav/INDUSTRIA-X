"""
backend/services/whatif_service.py
====================================
Phase 11: What-If Analysis Service

Accepts a baseline set of machine inputs and a scenario (modified parameters),
runs the existing ML pipeline on both, and returns a structured comparison.

DESIGN
------
- No new model is created. No model is retrained.
- The same assess_machine() function used by the assessment API is called twice:
  once for the baseline, once for the scenario.
- All calculations are deterministic. No LLM is used for numerical outputs.
- Scenario inputs are validated against documented training-data ranges.
  Values outside the observed range produce a warning; they are NOT rejected
  silently clamped — the caller is informed.

DISCLAIMER
----------
What-if results are model simulations based on the AI4I 2020 synthetic
training dataset. They are NOT guaranteed physical outcomes. Results should
be interpreted cautiously, especially for inputs outside the training
distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Training-data observed ranges (from ml/data/validation.py NUMERICAL_RANGES)
# Used ONLY to emit an out-of-distribution warning — inputs are NOT clamped.
_TRAINING_RANGES: dict[str, tuple[float, float]] = {
    "air_temp_k":              (295.0, 305.0),
    "process_temp_k":         (305.0, 315.0),
    "rotational_speed_rpm":   (1000.0, 3000.0),
    "torque_nm":              (0.0, 80.0),
    "tool_wear_min":          (0.0, 300.0),
}

# Hard-reject thresholds: values this far outside training ranges are
# considered physically impossible / extreme enough to refuse outright.
# Chosen to be 2× the range width beyond the boundary.
_EXTREME_MULTIPLIER: float = 2.0

_VALID_TYPES: frozenset[str] = frozenset({"L", "M", "H"})


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MachineInputs:
    """The six ML features accepted by the INDUSTRIA-X pipeline."""

    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    type_: str


@dataclass(frozen=True)
class AssessmentSnapshot:
    """Results from one ML pipeline run."""

    failure_probability: float
    anomaly_score: float
    risk_score: float
    risk_level: str


@dataclass(frozen=True)
class WhatIfResult:
    """Full what-if comparison response."""

    baseline: AssessmentSnapshot
    scenario: AssessmentSnapshot
    failure_probability_delta: float
    anomaly_score_delta: float
    risk_score_delta: float
    risk_level_changed: bool
    out_of_distribution_warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": {
                "failure_probability": self.baseline.failure_probability,
                "anomaly_score": self.baseline.anomaly_score,
                "risk_score": self.baseline.risk_score,
                "risk_level": self.baseline.risk_level,
            },
            "scenario": {
                "failure_probability": self.scenario.failure_probability,
                "anomaly_score": self.scenario.anomaly_score,
                "risk_score": self.scenario.risk_score,
                "risk_level": self.scenario.risk_level,
            },
            "change": {
                "failure_probability_delta": self.failure_probability_delta,
                "anomaly_score_delta": self.anomaly_score_delta,
                "risk_score_delta": self.risk_score_delta,
                "risk_level_changed": self.risk_level_changed,
            },
            "out_of_distribution_warning": self.out_of_distribution_warning,
        }


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_inputs(inputs: MachineInputs, label: str = "input") -> None:
    """Validate that all feature values are safe to pass to the ML pipeline.

    Raises ValueError for:
    - NaN or infinity
    - Impossible negative values (temperature, RPM, torque, tool wear)
    - Invalid Type
    - Extreme values: more than 2× the training range width beyond the boundary

    Parameters
    ----------
    inputs:
        The MachineInputs to validate.
    label:
        Human-readable label for error messages ('baseline' or 'scenario').
    """
    if inputs.type_ not in _VALID_TYPES:
        raise ValueError(
            f"{label}: 'type' must be one of {sorted(_VALID_TYPES)}, "
            f"got '{inputs.type_}'."
        )

    numeric = {
        "air_temp_k": inputs.air_temp_k,
        "process_temp_k": inputs.process_temp_k,
        "rotational_speed_rpm": inputs.rotational_speed_rpm,
        "torque_nm": inputs.torque_nm,
        "tool_wear_min": inputs.tool_wear_min,
    }

    for name, value in numeric.items():
        # NaN / infinity
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            raise ValueError(
                f"{label}: '{name}' must not be NaN or infinity, got {value!r}."
            )

        # Impossible negatives (temperatures must be positive Kelvin;
        # RPM, torque, tool wear are non-negative in this context)
        if value < 0:
            raise ValueError(
                f"{label}: '{name}' must be non-negative, got {value}."
            )

        # Extreme values beyond 2× training range width
        lo, hi = _TRAINING_RANGES[name]
        width = hi - lo
        extreme_lo = lo - _EXTREME_MULTIPLIER * width
        extreme_hi = hi + _EXTREME_MULTIPLIER * width
        if value < extreme_lo or value > extreme_hi:
            raise ValueError(
                f"{label}: '{name}' value {value} is extremely outside the "
                f"documented training-data range [{lo}, {hi}] and cannot be "
                f"reliably processed by the model."
            )


def _check_out_of_distribution(inputs: MachineInputs) -> list[str]:
    """Return a list of feature names that are outside the training range.

    Does NOT raise — just returns the names for a warning message.
    """
    ood: list[str] = []
    numeric = {
        "air_temp_k": inputs.air_temp_k,
        "process_temp_k": inputs.process_temp_k,
        "rotational_speed_rpm": inputs.rotational_speed_rpm,
        "torque_nm": inputs.torque_nm,
        "tool_wear_min": inputs.tool_wear_min,
    }
    for name, value in numeric.items():
        lo, hi = _TRAINING_RANGES[name]
        if not (lo <= value <= hi):
            ood.append(name)
    return ood


# ── Core service function ─────────────────────────────────────────────────────


def run_whatif(baseline: MachineInputs, scenario: MachineInputs) -> WhatIfResult:
    """Run a what-if comparison between a baseline and a scenario.

    Calls ``ml.assessment.assess_machine`` twice — once per input set —
    using the same unmodified ML pipeline used by the assessment API.

    Parameters
    ----------
    baseline:
        Baseline machine operating conditions.
    scenario:
        Modified operating conditions to compare against the baseline.

    Returns
    -------
    WhatIfResult
        Structured comparison with baseline results, scenario results,
        deltas, and an optional out-of-distribution warning.

    Raises
    ------
    ValueError
        If either input set fails validation.
    FileNotFoundError
        If ML artifacts are missing.
    RuntimeError
        If the ML pipeline fails unexpectedly.
    """
    from ml.assessment import assess_machine  # noqa: PLC0415

    # Validate both inputs before calling the ML pipeline
    _validate_inputs(baseline, "baseline")
    _validate_inputs(scenario, "scenario")

    # Check out-of-distribution for scenario (warn, do not reject)
    ood_fields = _check_out_of_distribution(scenario)
    ood_warning: str | None = None
    if ood_fields:
        ood_warning = (
            "Scenario is outside the observed training-data range for: "
            + ", ".join(ood_fields)
            + ". Results should be interpreted cautiously — the model was not "
            "trained on these operating conditions."
        )

    # Run ML pipeline on baseline
    try:
        baseline_ml = assess_machine(
            air_temp_k=baseline.air_temp_k,
            process_temp_k=baseline.process_temp_k,
            rotational_speed_rpm=baseline.rotational_speed_rpm,
            torque_nm=baseline.torque_nm,
            tool_wear_min=baseline.tool_wear_min,
            type_=baseline.type_,
        )
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ML pipeline failed on baseline: {exc}") from exc

    # Run ML pipeline on scenario
    try:
        scenario_ml = assess_machine(
            air_temp_k=scenario.air_temp_k,
            process_temp_k=scenario.process_temp_k,
            rotational_speed_rpm=scenario.rotational_speed_rpm,
            torque_nm=scenario.torque_nm,
            tool_wear_min=scenario.tool_wear_min,
            type_=scenario.type_,
        )
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ML pipeline failed on scenario: {exc}") from exc

    baseline_snap = AssessmentSnapshot(
        failure_probability=baseline_ml.failure_probability,
        anomaly_score=baseline_ml.anomaly_score,
        risk_score=baseline_ml.risk_score,
        risk_level=baseline_ml.risk_level,
    )
    scenario_snap = AssessmentSnapshot(
        failure_probability=scenario_ml.failure_probability,
        anomaly_score=scenario_ml.anomaly_score,
        risk_score=scenario_ml.risk_score,
        risk_level=scenario_ml.risk_level,
    )

    return WhatIfResult(
        baseline=baseline_snap,
        scenario=scenario_snap,
        failure_probability_delta=round(
            scenario_ml.failure_probability - baseline_ml.failure_probability, 6
        ),
        anomaly_score_delta=round(
            scenario_ml.anomaly_score - baseline_ml.anomaly_score, 6
        ),
        risk_score_delta=round(
            scenario_ml.risk_score - baseline_ml.risk_score, 6
        ),
        risk_level_changed=(baseline_ml.risk_level != scenario_ml.risk_level),
        out_of_distribution_warning=ood_warning,
    )
