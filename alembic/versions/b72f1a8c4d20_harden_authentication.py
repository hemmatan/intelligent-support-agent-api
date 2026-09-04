"""harden authentication

Revision ID: b72f1a8c4d20
Revises: a8c94d2f2887
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b72f1a8c4d20"
down_revision: str | None = "a8c94d2f2887"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add roles and revocable hashed bearer-token storage."""
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "role", sa.String(length=32), nullable=False, server_default="customer"
            )
        )
        batch_op.add_column(
            sa.Column(
                "preferred_locale",
                sa.String(length=5),
                nullable=False,
                server_default="en",
            )
        )
        batch_op.add_column(
            sa.Column("external_customer_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "is_active", sa.Boolean(), nullable=False, server_default=sa.true()
            )
        )
        batch_op.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch_op.create_unique_constraint(
            "uq_users_external_customer_id", ["external_customer_id"]
        )

    # Existing API tokens were stored in plaintext and must not remain valid.
    op.execute(sa.text("DELETE FROM api_tokens"))
    op.drop_index("ix_api_tokens_token", table_name="api_tokens")
    with op.batch_alter_table("api_tokens") as batch_op:
        batch_op.alter_column(
            "token",
            new_column_name="token_hash",
            existing_type=sa.String(),
            type_=sa.String(length=64),
            existing_nullable=False,
        )
        batch_op.add_column(
            sa.Column(
                "name", sa.String(length=80), nullable=False, server_default="default"
            )
        )
        batch_op.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch_op.add_column(sa.Column("expires_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("revoked_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("last_used_at", sa.DateTime(timezone=True)))
        batch_op.create_index("ix_api_tokens_user_id", ["user_id"])
    op.create_index(
        "ix_api_tokens_token_hash", "api_tokens", ["token_hash"], unique=True
    )

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("jti", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("replaced_by_jti", sa.String(length=36)),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # jti uniqueness comes from ix_refresh_tokens_jti below.
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_refresh_tokens_jti", "refresh_tokens", ["jti"], unique=True)
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])


def downgrade() -> None:
    """Restore the starter authentication schema."""
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_jti", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    # Mirror of the upgrade: the stored digests would otherwise be renamed into
    # the plaintext column, where the previous application accepts the digest
    # itself as a working token.
    op.execute(sa.text("DELETE FROM api_tokens"))
    op.drop_index("ix_api_tokens_token_hash", table_name="api_tokens")
    with op.batch_alter_table("api_tokens") as batch_op:
        batch_op.drop_index("ix_api_tokens_user_id")
        batch_op.drop_column("last_used_at")
        batch_op.drop_column("revoked_at")
        batch_op.drop_column("expires_at")
        batch_op.drop_column("created_at")
        batch_op.drop_column("name")
        batch_op.alter_column(
            "token_hash",
            new_column_name="token",
            existing_type=sa.String(length=64),
            type_=sa.String(),
            existing_nullable=False,
        )
    op.create_index("ix_api_tokens_token", "api_tokens", ["token"], unique=True)

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("uq_users_external_customer_id", type_="unique")
        batch_op.drop_column("updated_at")
        batch_op.drop_column("created_at")
        batch_op.drop_column("is_active")
        batch_op.drop_column("external_customer_id")
        batch_op.drop_column("preferred_locale")
        batch_op.drop_column("role")
