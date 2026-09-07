"""record who finished a case

Revision ID: cb8428457871
Revises: 2055d978700c
Create Date: 2026-09-06 18:02:26.029280

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cb8428457871"
down_revision: str | None = "2055d978700c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    Two passes in batch mode, since SQLite rebuilds the table for each and a
    column cannot be added and referenced in the same rebuild.
    """
    with op.batch_alter_table("support_cases") as batch:
        batch.add_column(sa.Column("resolved_by", sa.Integer(), nullable=True))

    with op.batch_alter_table("support_cases") as batch:
        batch.create_foreign_key(
            "fk_support_cases_resolved_by_users",
            "users",
            ["resolved_by"],
            ["id"],
            ondelete="SET NULL",
        )

    # Cases closed before this column existed were finished by whoever had
    # taken them on, which was the only name recorded at the time.
    op.execute(
        "UPDATE support_cases SET resolved_by = assigned_to WHERE closed_at IS NOT NULL"
    )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("support_cases") as batch:
        batch.drop_constraint("fk_support_cases_resolved_by_users", type_="foreignkey")
        batch.drop_column("resolved_by")
