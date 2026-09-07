"""keep the source plan a request was decided under

Revision ID: 64648de6b07a
Revises: cb8428457871
Create Date: 2026-09-06 18:06:01.301314

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "64648de6b07a"
down_revision: str | None = "cb8428457871"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    An empty list for rows already present. What those requests were allowed
    to read is not recoverable now, and inventing it from today's profiles
    would put a guess in an audit record.
    """
    with op.batch_alter_table("support_cases") as batch:
        batch.add_column(
            sa.Column("sources", sa.JSON(), nullable=False, server_default="[]")
        )

    with op.batch_alter_table("support_cases") as batch:
        batch.alter_column("sources", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("support_cases") as batch:
        batch.drop_column("sources")
