"""Normalising text before matching it.

Shared so that retrieval and risk detection agree about what two strings
being the same means. Two copies of this would drift, and the drift would
show up as a French message matching in one place and not the other.
"""

import unicodedata
from collections.abc import Mapping

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


def rendered(field: str, value: object, words: Mapping[str, str], joiner: str) -> str:
    """One claim value as a particular language writes it.

    Figures print as themselves. Everything named — a token, a choice, a
    yes-or-no — goes through the wording, because printing it raw puts an
    internal value in front of somebody: in the searchable text it matches
    nothing anybody types, and in a reply it is gibberish.

    Shared between the two surfaces on purpose. They read from different
    wordings, the searchable one and the approved one, and the rule for
    turning a value into language is the same rule; written twice it would be
    two rules that agree until one of them is edited.
    """
    if isinstance(value, list | tuple):
        spoken = [rendered(field, item, words, joiner) for item in value]
        if len(spoken) < 2:
            return "".join(spoken)
        return f"{', '.join(spoken[:-1])} {joiner} {spoken[-1]}"
    # bool before str and int: True is an int, and "True" is not an answer.
    # Keyed by the field, because two yes-or-no claims in one entry would
    # otherwise both want the key "true" and one would silently win.
    if isinstance(value, bool):
        return words[f"{field}_{str(value).lower()}"]
    if isinstance(value, str):
        return words[value]
    return str(value)
