"""
backend/models/machine.py
=========================
SQLAlchemy ORM model for Machine identity and metadata.

Phase 8: persistence layer — stores machine identity so assessments
can be associated with a specific machine identifier.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base

if TYPE_CHECKING:
    from backend.models.machine_assessment import MachineAssessment


class Machine(Base):
    """Represents a physical or virtual industrial machine."""

    __tablename__ = "machines"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    machine_identifier: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    assessments: Mapped[list[MachineAssessment]] = relationship(
        "MachineAssessment",
        back_populates="machine",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Machine id={self.id} identifier={self.machine_identifier!r}>"
