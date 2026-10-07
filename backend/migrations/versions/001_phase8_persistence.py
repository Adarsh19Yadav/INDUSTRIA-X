"""Phase 8 — create machines, machine_assessments, agent_audit_logs tables.

Revision ID: 001_phase8_persistence
Revises: (none — first migration)
Create Date: 2025-01-01 00:00:00.000000

This migration is reproducible from a clean database.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision: str = "001_phase8_persistence"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── machines ─────────────────────────────────────────────────────────────
    op.create_table(
        "machines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("machine_identifier", sa.String(128), nullable=False),
        sa.Column("type", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_machines_id"), "machines", ["id"], unique=False)
    op.create_index(
        op.f("ix_machines_machine_identifier"),
        "machines",
        ["machine_identifier"],
        unique=True,
    )

    # ── machine_assessments ───────────────────────────────────────────────────
    op.create_table(
        "machine_assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("machine_id", sa.Integer(), nullable=False),
        sa.Column("failure_probability", sa.Float(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["machine_id"], ["machines.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_machine_assessments_id"), "machine_assessments", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_machine_assessments_machine_id"),
        "machine_assessments",
        ["machine_id"],
        unique=False,
    )

    # ── agent_audit_logs ──────────────────────────────────────────────────────
    op.create_table(
        "agent_audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("query", sa.String(1000), nullable=False),
        sa.Column("machine_id", sa.String(128), nullable=True),
        sa.Column("tools_used", sa.String(512), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("response_summary", sa.String(1000), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_agent_audit_logs_id"), "agent_audit_logs", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_agent_audit_logs_request_id"),
        "agent_audit_logs",
        ["request_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_agent_audit_logs_session_id"),
        "agent_audit_logs",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_audit_logs_created_at"),
        "agent_audit_logs",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_audit_logs_created_at"), table_name="agent_audit_logs")
    op.drop_index(op.f("ix_agent_audit_logs_session_id"), table_name="agent_audit_logs")
    op.drop_index(op.f("ix_agent_audit_logs_request_id"), table_name="agent_audit_logs")
    op.drop_index(op.f("ix_agent_audit_logs_id"), table_name="agent_audit_logs")
    op.drop_table("agent_audit_logs")

    op.drop_index(op.f("ix_machine_assessments_machine_id"), table_name="machine_assessments")
    op.drop_index(op.f("ix_machine_assessments_id"), table_name="machine_assessments")
    op.drop_table("machine_assessments")

    op.drop_index(op.f("ix_machines_machine_identifier"), table_name="machines")
    op.drop_index(op.f("ix_machines_id"), table_name="machines")
    op.drop_table("machines")
