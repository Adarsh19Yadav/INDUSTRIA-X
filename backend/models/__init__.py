"""backend/models/__init__.py — export all ORM models.

Importing this package makes all models visible to Alembic's
autogenerate so that migrations detect all tables correctly.
"""

from backend.models.machine import Machine  # noqa: F401
from backend.models.machine_assessment import MachineAssessment  # noqa: F401
from backend.models.agent_audit_log import AgentAuditLog  # noqa: F401

__all__ = ["Machine", "MachineAssessment", "AgentAuditLog"]
