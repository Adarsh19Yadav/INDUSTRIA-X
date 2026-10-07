"""ml/anomaly_inference.py — Phase 4 anomaly detection inference interface.

This module provides a single public function ``detect_anomaly`` that accepts
one machine observation and returns an anomaly score, normalised anomaly
severity, and a binary anomaly flag.

ANOMALY SCORE DEFINITION
------------------------
Isolation Forest's ``decision_function`` returns a raw score that is:
  - Positive  → more normal (easy to isolate requires many splits)
  - Negative  → more anomalous (easy to isolate requires few splits)
  - ≈ 0       → near the contamination-implied decision boundary

This raw score is NOT a probability.

To produce an intuitive ``anomaly_score`` in [0, 1] where 1 = most anomalous:

    normalised = clip( (−raw_score − min_bound) / (max_bound − min_bound), 0, 1)

where ``min_bound`` and ``max_bound`` are derived from the training distribution
stored in ``anomaly_metadata.json``:
  min_bound = −train_score_max   (most normal train score, negated)
  max_bound = −train_score_min   (most anomalous train score, negated)

An ``anomaly_score`` close to 1 means the observation is as anomalous as the
most extreme point seen during training.  An ``anomaly_score`` close to 0
means the observation is firmly within the normal training distribution.

IS_ANOMALY FLAG
---------------
``is_anomaly = True`` when the raw decision_function score < 0 (i.e., the
IsolationForest classifier labels the observation as an outlier).  This is
the sklearn-native boundary set by the fitted contamination parameter.

IMPORTANT
---------
Anomaly detection identifies unusual operating patterns.  It does NOT prove
that a machine is failing.  A high anomaly_score requires investigation, not
an automatic shutdown.

Usage
-----
    from ml.anomaly_inference import detect_anomaly

    result = detect_anomaly(
        air_temp_k=300.5,
        process_temp_k=310.2,
        rotational_speed_rpm=1500,
        torque_nm=40.0,
        tool_wear_min=120,
        type_="M",
    )
    print(result)
    # AnomalyResult(anomaly_score=0.12, is_anomaly=False, raw_score=0.043,
    #               anomaly_version='v1')

SYNTHETIC DATA NOTICE
---------------------
The underlying detector was trained on the AI4I 2020 *synthetic* dataset.
It has NOT been validated on real factory equipment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
class AnomalyResult:
    """Return type of :func:`detect_anomaly`.

    Attributes
    ----------
    anomaly_score:
        Normalised score in [0, 1].  1 = maximally anomalous relative to
        the training distribution.  0 = firmly in the normal range.
        This is NOT a probability.
    is_anomaly:
        True when the observation falls outside the IsolationForest decision
        boundary (raw_score < 0).  Determined by the contamination-based
        threshold fitted on training data.
    raw_score:
        The raw output of sklearn's ``decision_function``.  Positive = inlier,
        negative = outlier.  Provided for auditability.
    anomaly_version:
        Artifact version string (e.g. ``'v1'``).
    """

    anomaly_score: float
    is_anomaly: bool
    raw_score: float
    anomaly_version: str

    def to_dict(self) -> dict:
        return {
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "raw_score": self.raw_score,
            "anomaly_version": self.anomaly_version,
        }


# ---------------------------------------------------------------------------
# Artifact loader (singleton cache)
# ---------------------------------------------------------------------------


class _AnomalyCache:
    """Lazy-loading singleton for the trained anomaly pipeline and metadata."""

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
            pipeline_path = target_dir / "anomaly_pipeline.joblib"
            metadata_path = target_dir / "anomaly_metadata.json"

            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Anomaly pipeline not found at '{pipeline_path}'.\n"
                    "Run 'python -m ml.anomaly_train' to generate the artifact."
                )
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Anomaly metadata not found at '{metadata_path}'.\n"
                    "Run 'python -m ml.anomaly_train' to generate the artifact."
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
# Score normalisation
# ---------------------------------------------------------------------------


def _normalise_score(raw_score: float, train_score_min: float, train_score_max: float) -> float:
    """Map a raw IsolationForest decision_function score to [0, 1].

    Transformation
    --------------
    The raw decision_function score is positive for normal and negative for
    anomalous observations.  We negate it so that higher values = more
    anomalous, then clip to the range observed in training data.

    Parameters
    ----------
    raw_score:
        Output of ``pipeline.decision_function(X)[0]``.
    train_score_min:
        Minimum decision_function score on training data (most anomalous
        training observation).
    train_score_max:
        Maximum decision_function score on training data (most normal
        training observation).

    Returns
    -------
    float
        Normalised anomaly score in [0, 1].
    """
    # Negate: high value = anomalous
    negated = -raw_score
    min_bound = -train_score_max  # negated most-normal → becomes lowest anomaly
    max_bound = -train_score_min  # negated most-anomalous → becomes highest anomaly

    denom = max_bound - min_bound
    if denom <= 0:
        # Degenerate: all training scores identical (should not happen in practice)
        return 0.0

    normalised = (negated - min_bound) / denom
    return float(np.clip(normalised, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomaly(
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    type_: str,
    *,
    artifacts_dir: str | Path | None = None,
) -> AnomalyResult:
    """Detect whether one machine observation is anomalous.

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
    AnomalyResult
        Dataclass with ``anomaly_score``, ``is_anomaly``, ``raw_score``,
        and ``anomaly_version``.

    Raises
    ------
    ValueError
        If ``type_`` is not in ``{'L', 'M', 'H'}`` or if any numeric input
        is ``None`` or ``NaN``.
    FileNotFoundError
        If the anomaly artifact files are not found.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if type_ not in _VALID_TYPES:
        raise ValueError(
            f"Invalid 'type_' value '{type_}'. Must be one of {sorted(_VALID_TYPES)}."
        )

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

    # ------------------------------------------------------------------
    # Load artifact
    # ------------------------------------------------------------------
    target_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    pipeline, metadata = _AnomalyCache.load(target_dir)

    # ------------------------------------------------------------------
    # Build observation dataframe
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    raw_score = float(pipeline.decision_function(observation)[0])
    is_anomaly = bool(pipeline.predict(observation)[0] == -1)

    anomaly_score = _normalise_score(
        raw_score,
        train_score_min=float(metadata["train_score_min"]),
        train_score_max=float(metadata["train_score_max"]),
    )
    anomaly_version = str(metadata["anomaly_version"])

    return AnomalyResult(
        anomaly_score=round(anomaly_score, 6),
        is_anomaly=is_anomaly,
        raw_score=round(raw_score, 6),
        anomaly_version=anomaly_version,
    )


def get_anomaly_metadata(artifacts_dir: str | Path | None = None) -> dict:
    """Return the anomaly model metadata dictionary.

    Parameters
    ----------
    artifacts_dir:
        Optional override for the artifacts directory.

    Returns
    -------
    dict
        Full metadata as stored in ``anomaly_metadata.json``.
    """
    target_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    _, metadata = _AnomalyCache.load(target_dir)
    return metadata
