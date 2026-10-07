"""ml/inference.py — Reusable inference interface for the Phase 3 failure model.

This module provides a single public function ``predict_failure`` that accepts
one machine observation and returns failure probability, predicted class,
threshold, and model version.

It is designed to be consumed by the INDUSTRIA-X API and agent tools.

Usage
-----
    from ml.inference import predict_failure

    result = predict_failure(
        air_temp_k=300.5,
        process_temp_k=310.2,
        rotational_speed_rpm=1500,
        torque_nm=40.0,
        tool_wear_min=120,
        type_="M",
    )
    print(result)
    # InferenceResult(failure_probability=0.31, predicted_failure=0,
    #                 threshold=0.543, model_version='v1')

SYNTHETIC DATA NOTICE
---------------------
The underlying model was trained on the AI4I 2020 *synthetic* dataset.
It has NOT been validated on real factory equipment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd

from ml.data.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERICAL_FEATURES

# ---------------------------------------------------------------------------
# Default artifact directory
# ---------------------------------------------------------------------------

_DEFAULT_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "v1"

# ---------------------------------------------------------------------------
# Valid categories
# ---------------------------------------------------------------------------

_VALID_TYPES: frozenset[str] = frozenset({"L", "M", "H"})

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """Return type of :func:`predict_failure`."""

    failure_probability: float
    predicted_failure: int
    threshold: float
    model_version: str

    def to_dict(self) -> dict:
        return {
            "failure_probability": self.failure_probability,
            "predicted_failure": self.predicted_failure,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Model loader (singleton cache)
# ---------------------------------------------------------------------------


class _ModelCache:
    """Lazy-loading singleton for the trained pipeline and metadata."""

    _pipeline = None
    _metadata: dict | None = None
    _artifacts_dir: Path | None = None

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> tuple:
        """Return (pipeline, metadata) from cache or disk.

        Parameters
        ----------
        artifacts_dir:
            Override the default artifact directory.  Passing a non-``None``
            value forces a reload from that directory.
        """
        target_dir = artifacts_dir or _DEFAULT_ARTIFACTS_DIR

        if cls._pipeline is None or cls._artifacts_dir != target_dir:
            pipeline_path = target_dir / "model_pipeline.joblib"
            metadata_path = target_dir / "metadata.json"

            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Model pipeline not found at '{pipeline_path}'.\n"
                    "Run 'python -m ml.models.train' to generate the artifact."
                )
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Model metadata not found at '{metadata_path}'.\n"
                    "Run 'python -m ml.models.train' to generate the artifact."
                )

            cls._pipeline = joblib.load(pipeline_path)
            with open(metadata_path, encoding="utf-8") as f:
                cls._metadata = json.load(f)
            cls._artifacts_dir = target_dir

        return cls._pipeline, cls._metadata

    @classmethod
    def clear_cache(cls) -> None:
        """Force re-load on the next call (used in tests)."""
        cls._pipeline = None
        cls._metadata = None
        cls._artifacts_dir = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_failure(
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    type_: str,
    *,
    artifacts_dir: str | Path | None = None,
) -> InferenceResult:
    """Predict machine failure for one observation.

    Parameters
    ----------
    air_temp_k:
        Air temperature in Kelvin.  Expected range: ~295–305 K.
    process_temp_k:
        Process temperature in Kelvin.  Expected range: ~305–315 K.
    rotational_speed_rpm:
        Rotational speed in RPM.  Expected range: ~1100–2900 rpm.
    torque_nm:
        Torque in Newton-metres.  Expected range: ~3–77 Nm.
    tool_wear_min:
        Cumulative tool wear in minutes.  Expected range: 0–254 min.
    type_:
        Product quality variant.  Must be one of ``'L'``, ``'M'``, ``'H'``.
    artifacts_dir:
        Optional override for the artifacts directory.
        Defaults to ``ml/artifacts/v1``.

    Returns
    -------
    InferenceResult
        Dataclass with ``failure_probability``, ``predicted_failure``,
        ``threshold``, and ``model_version``.

    Raises
    ------
    ValueError
        If ``type_`` is not one of ``'L'``, ``'M'``, ``'H'``.
    KeyError
        If the feature set in the saved metadata does not match FEATURE_COLUMNS.
    FileNotFoundError
        If the model artifact files are not found.
    """
    # Validate categorical
    if type_ not in _VALID_TYPES:
        raise ValueError(
            f"Invalid 'type_' value '{type_}'. Must be one of {sorted(_VALID_TYPES)}."
        )

    # Validate numeric inputs are not NaN/None
    numeric_inputs = {
        "air_temp_k": air_temp_k,
        "process_temp_k": process_temp_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }
    for name, val in numeric_inputs.items():
        if val is None:
            raise ValueError(f"Feature '{name}' must not be None or NaN.")
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            raise ValueError(f"Feature '{name}' must be a finite number, got {val}.")

    # Load model
    target_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    pipeline, metadata = _ModelCache.load(target_dir)

    # Validate feature columns match
    expected_features = metadata.get("feature_columns", [])
    if expected_features and expected_features != FEATURE_COLUMNS:
        raise KeyError(
            f"Metadata feature_columns {expected_features} do not match "
            f"current FEATURE_COLUMNS {FEATURE_COLUMNS}."
        )

    threshold = float(metadata["threshold"])
    model_version = str(metadata["model_version"])

    # Build observation dataframe in the correct column order
    observation = pd.DataFrame(
        [
            {
                "Air temperature [K]": float(air_temp_k),
                "Process temperature [K]": float(process_temp_k),
                "Rotational speed [rpm]": float(rotational_speed_rpm),
                "Torque [Nm]": float(torque_nm),
                "Tool wear [min]": float(tool_wear_min),
                "Type": str(type_),
            }
        ]
    )[FEATURE_COLUMNS]  # enforce column order

    proba = float(pipeline.predict_proba(observation)[0, 1])
    predicted = int(proba >= threshold)

    return InferenceResult(
        failure_probability=round(proba, 6),
        predicted_failure=predicted,
        threshold=threshold,
        model_version=model_version,
    )


def get_model_metadata(artifacts_dir: str | Path | None = None) -> dict:
    """Return the model metadata dictionary.

    Parameters
    ----------
    artifacts_dir:
        Optional override for the artifacts directory.

    Returns
    -------
    dict
        Full metadata as stored in ``metadata.json``.
    """
    target_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    _, metadata = _ModelCache.load(target_dir)
    return metadata
