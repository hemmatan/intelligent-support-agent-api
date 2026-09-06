"""What the schema changes do to rows that already exist.

Applying a migration to an empty table proves it parses. It says nothing about
the data, and the only interesting question about a migration is what happens
to what is already there.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

BEFORE = "6b201e0cc081"
AFTER = "2055d978700c"
ANSWER = "Returns are accepted within 30 days of delivery."


@pytest.fixture
def migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Config]:
    """Alembic pointed at a database of this test's own."""
    from app.core.config import settings

    database = tmp_path / "migration.sqlite3"
    monkeypatch.setattr(settings, "DB_NAME", str(database))
    config = Config("alembic.ini")
    yield config
    command.downgrade(config, "base")


def a_case_answered_before_the_change(database: Path) -> None:
    """One row in the shape the older schema held."""
    db = sqlite3.connect(database)
    db.execute(
        "INSERT INTO users (username, hashed_password, role, preferred_locale,"
        " is_active, created_at, updated_at)"
        " VALUES ('someone','x','customer','en',1,'2026-09-01','2026-09-01')"
    )
    db.execute(
        "INSERT INTO support_cases (reference, user_id, message, locale, route,"
        " reasons, intent, reply, citations, wording, reliability, created_at)"
        " VALUES ('older', 1, 'How long do I have to return a jacket?', 'en',"
        " 'direct_response', '[]', 'return_policy', ?, '[]', '[]', NULL,"
        " '2026-09-01')",
        (ANSWER,),
    )
    db.commit()
    db.close()


def value_of(database: Path, column: str) -> str | None:
    db = sqlite3.connect(database)
    try:
        row = db.execute(
            f"SELECT {column} FROM support_cases WHERE reference='older'"  # noqa: S608
        ).fetchone()
        return None if row is None else row[0]
    finally:
        db.close()


def test_an_answer_already_sent_survives_the_column_being_replaced(
    migrated: Config, tmp_path: Path
) -> None:
    """The reply column was dropped and its contents were not carried across.

    An empty table said the migration worked, and so did a row whose reply was
    null — which is every row except the ones this question is about. What a
    customer was told is the thing a case exists to record, and it was being
    replaced with an empty string while the upgrade reported success.
    """
    database = tmp_path / "migration.sqlite3"
    command.upgrade(migrated, BEFORE)
    a_case_answered_before_the_change(database)
    assert value_of(database, "reply") == ANSWER

    command.upgrade(migrated, AFTER)
    assert value_of(database, "sent") == ANSWER


def test_the_words_survive_going_back_as_well(migrated: Config, tmp_path: Path) -> None:
    """Reversing is for getting out of trouble, not for adding to it."""
    database = tmp_path / "migration.sqlite3"
    command.upgrade(migrated, BEFORE)
    a_case_answered_before_the_change(database)
    command.upgrade(migrated, AFTER)

    command.downgrade(migrated, BEFORE)
    assert value_of(database, "reply") == ANSWER
