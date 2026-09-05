"""Normalising text before matching it.

Shared so that retrieval and risk detection agree about what two strings
being the same means. Two copies of this would drift, and the drift would
show up as a French message matching in one place and not the other.
"""

import unicodedata

# A typographic apostrophe and a typed one are the same character to a reader,
# and "didn't" reporting an unrecognised purchase must not depend on which
# keyboard produced it. They are normalised rather than removed: dropping them
# would glue "d'origine" into one word and change what retrieval indexes.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


def fold(text: str) -> str:
    """Lowercase and strip accents, leaving everything else alone.

    Customers type "hygiene" for "hygiène" and "debite" for "débité", and
    their keyboard decides which apostrophe lands in "didn't". None of that
    should decide what happens next.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.translate(_APOSTROPHES)
