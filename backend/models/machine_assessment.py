"""
backend/models/machine_assessment.py
=====================================
SQLAlchemy ORM model for persisting ML pipeline assessment results.

Phase 8: stores the output of ml.assessment.assess_machine() so that
assessment history is queryable per machine.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base

if TYPE_CHECKING:
    from backend.models.machine import Machine


class MachineAssessment(Base):
    """Persisted result of a single ML assessment run for a machine."""

    __tablename__ = "machine_assessments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    machine_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # ML output fields — sourced exclusively from ml.assessment.assess_machine()
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    machine: Mapped[Machine] = relationship("Machine", back_populates="assessments")

    def __repr__(self) -> str:
        return (
            f"<MachineAssessment id={self.id} machine_id={self.machine_id} "
            f"risk_level={self.risk_level!r}>"
        )
