"""The record every support request leaves behind.

Written before a reply is returned, whatever the reply turns out to be. A
response that reached somebody with no record of why is the one case that
cannot be looked into afterwards, and support systems are looked into
afterwards by definition.

It is also the queue. Telling a customer their message has gone to a
colleague, and then not putting it anywhere a colleague looks, is a false
statement about what happened — so the rows a person works are the rows this
table holds with an escalating or reviewing route and no outcome yet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.user import User, utc_now


class SupportCase(Base):
    """One request, the decision taken about it, and what it rested on."""

    __tablename__ = "support_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Minted before the reply is rendered, so the customer can quote it and
    # one insert covers the whole request.
    reference: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    message: Mapped[str] = mapped_column(Text, nullable=False)
    locale: Mapped[str] = mapped_column(String(5), nullable=False)

    # What the customer gave us, kept rather than counted. Somebody picking
    # this up needs the order number, not the knowledge that one was supplied,
    # and going back to the customer for it is the whole thing this avoids.
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    product_reference: Mapped[str | None] = mapped_column(String(64), nullable=True)
    external_customer_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    route: Mapped[str] = mapped_column(String(32), nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # The words that reached them, whichever of the four outcomes it was.
    # Kept verbatim rather than rebuilt from the reference on demand, because
    # approved wording gets edited and the record has to say what was sent,
    # not what would be sent today.
    sent: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    wording: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    reliability: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    # Who has it, and what they did. Null assignment means nobody has picked
    # it up; null closure means it is still outstanding, which is what makes
    # this table a queue rather than a log.
    assigned_to: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(foreign_keys=[user_id])
    agent: Mapped[User | None] = relationship(foreign_keys=[assigned_to])

    __table_args__ = (Index("ix_support_cases_open", "route", "closed_at"),)
