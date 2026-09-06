"""keep the context a case was decided on

Revision ID: 2055d978700c
Revises: 6b201e0cc081
Create Date: 2026-09-06 17:36:58.506169

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2055d978700c"
down_revision: str | None = "6b201e0cc081"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    Batch mode throughout: SQLite cannot alter a constraint in place, so the
    table is copied. Postgres ignores the batching and issues plain ALTERs.

    `sent` arrives with a default so the statement succeeds against a table
    that already holds rows — a deployment could have served traffic between
    this revision and the last — and the default is dropped straight after,
    because a case that says nothing was sent is not a case.
    """
    with op.batch_alter_table("support_cases") as batch:
        batch.add_column(sa.Column("order_id", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("product_reference", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("external_customer_id", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("sent", sa.Text(), nullable=False, server_default="")
        )
        batch.add_column(sa.Column("assigned_to", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("resolution", sa.Text(), nullable=True))

    # Separate passes: batch mode rebuilds the table for each, and adding a
    # column, adding a constraint and dropping a column cannot be ordered
    # against one another in a single rebuild.
    with op.batch_alter_table("support_cases") as batch:
        batch.create_foreign_key(
            "fk_support_cases_assigned_to_users",
            "users",
            ["assigned_to"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("support_cases") as batch:
        batch.alter_column("sent", server_default=None)
        batch.drop_column("reply")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("support_cases") as batch:
        batch.add_column(sa.Column("reply", sa.TEXT(), nullable=True))
        batch.drop_constraint("fk_support_cases_assigned_to_users", type_="foreignkey")
        batch.drop_column("resolution")
        batch.drop_column("assigned_to")
        batch.drop_column("sent")
        batch.drop_column("external_customer_id")
        batch.drop_column("product_reference")
        batch.drop_column("order_id")
