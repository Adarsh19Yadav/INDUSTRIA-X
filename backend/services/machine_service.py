"""
backend/services/machine_service.py
=====================================
Service layer for Machine CRUD operations.

Phase 8: thin service so API endpoints remain clean and testable.
All DB access goes through SQLAlchemy sessions — no raw SQL.
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.machine import Machine
from backend.schemas.machine import MachineCreate

logger = logging.getLogger(__name__)


def create_machine(db: Session, data: MachineCreate) -> Machine:
    """Persist a new Machine.

    Raises
    ------
    ValueError
        If a machine with the same identifier already exists.
    """
    machine = Machine(
        machine_identifier=data.machine_identifier,
        type=data.type,
    )
    db.add(machine)
    try:
        db.commit()
        db.refresh(machine)
    except IntegrityError:
        db.rollback()
        raise ValueError(
            f"Machine with identifier '{data.machine_identifier}' already exists."
        )
    logger.info("[MachineService] Created machine id=%d ident=%r", machine.id, machine.machine_identifier)
    return machine


def get_machine_by_id(db: Session, machine_id: int) -> Machine | None:
    """Return a Machine by its integer primary key, or None."""
    return db.get(Machine, machine_id)


def get_machine_by_identifier(db: Session, identifier: str) -> Machine | None:
    """Return a Machine by its business identifier, or None."""
    return db.query(Machine).filter(Machine.machine_identifier == identifier).first()


def list_machines(db: Session, skip: int = 0, limit: int = 100) -> list[Machine]:
    """Return a paginated list of all machines."""
    return db.query(Machine).offset(skip).limit(limit).all()
