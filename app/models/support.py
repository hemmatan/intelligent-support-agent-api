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

    route: Mapped[str] = mapped_column(String(32), nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # What the customer received, and everything it was assembled from. Kept
    # verbatim rather than rebuilt on demand: approved wording gets edited,
    # and the record has to say what was sent, not what would be sent now.
    reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    wording: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    reliability: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    # Null while somebody still has to look at it, which is what makes this
    # table a queue rather than a log.
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped[User] = relationship()

    __table_args__ = (Index("ix_support_cases_open", "route", "closed_at"),)
