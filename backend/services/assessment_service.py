"""
backend/services/assessment_service.py
========================================
Service layer for machine assessments.

Phase 8: orchestrates the ML pipeline call and persists the result.
The ML logic itself lives entirely in ml/assessment.py — this module
only handles the persistence side.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.models.machine_assessment import MachineAssessment
from backend.schemas.machine import AssessmentRequest

logger = logging.getLogger(__name__)


def run_and_persist_assessment(
    db: Session,
    machine_id: int,
    req: AssessmentRequest,
) -> MachineAssessment:
    """Call the ML pipeline and persist the result.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    machine_id:
        Primary key of the Machine to associate with.
    req:
        Validated sensor reading inputs.

    Returns
    -------
    MachineAssessment
        The newly created and committed assessment record.

    Raises
    ------
    FileNotFoundError
        If an ML artifact is missing.
    ValueError
        If input validation inside the ML pipeline fails.
    RuntimeError
        On any unexpected ML-pipeline failure.
    """
    # Import here to keep the import cost off module-load
    from ml.assessment import assess_machine  # noqa: PLC0415

    try:
        result = assess_machine(
            air_temp_k=req.air_temperature_k,
            process_temp_k=req.process_temperature_k,
            rotational_speed_rpm=req.rotational_speed_rpm,
            torque_nm=req.torque_nm,
            tool_wear_min=req.tool_wear_min,
            type_=req.type_,
        )
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ML pipeline failed: {exc}") from exc

    assessment = MachineAssessment(
        machine_id=machine_id,
        failure_probability=result.failure_probability,
        anomaly_score=result.anomaly_score,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        model_version=result.model_version,
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    logger.info(
        "[AssessmentService] Persisted assessment id=%d machine_id=%d risk=%s",
        assessment.id,
        machine_id,
        result.risk_level,
    )
    return assessment


def list_assessments(
    db: Session,
    machine_id: int,
    skip: int = 0,
    limit: int = 50,
) -> list[MachineAssessment]:
    """Return paginated assessment history for a machine."""
    return (
        db.query(MachineAssessment)
        .filter(MachineAssessment.machine_id == machine_id)
        .order_by(MachineAssessment.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
