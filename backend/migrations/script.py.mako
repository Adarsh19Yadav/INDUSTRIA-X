"""Alembic migration script template — auto-generated."""
from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


# revision identifiers, used by Alembic.
revision: str = "${revision}"
down_revision: str | None = ${down_revision}
branch_labels: str | tuple[str, ...] | None = ${branch_labels}
depends_on: str | tuple[str, ...] | None = ${depends_on}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
