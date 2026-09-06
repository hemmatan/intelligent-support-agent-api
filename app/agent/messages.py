"""What a customer is told when no answer is being sent.

Being asked a question, handed to somebody, or held back are three of the four
things that can happen to a request, and until now all three arrived as a code
and nothing else. A code is not something to show anybody, so the customer got
silence — which the design says is not one of the outcomes.

These are not the sentences that answer questions. Those carry figures out of
approved claims and live in `responses`. These carry no facts at all: they say
what is happening to a request, so they take no slots and state no numbers.

Reason codes and wording are kept apart on purpose. A code is stable, and a
client branches on it, counts it and asserts against it. A sentence is written
for a person and gets translated. Several codes can share one sentence, and
usually should: what stopped a request is often not the customer's business,
and a queue entry saying which rating fell short would tell them nothing they
could act on.
"""

import hashlib
import json
import re
import tomllib
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    ValidationError,
)

from app.agent.knowledge import Locale
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    ReasonCode,
)
from app.agent.reliability import Route
from app.agent.text import fold

DEFAULT_MESSAGE_DIR = Path(__file__).parent / "outcome_messages"

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_PLACEHOLDER = re.compile(r"\{")
# Wordings that promise something nobody has checked. Digits are refused for
# the reason they are refused in policy prose; these are refused because a
# sentence with no figure in it can still commit the business to a timescale.
# It is a guard, not a substitute for somebody approving the text.
_PROMISES = (
    "soon",
    "shortly",
    "immediately",
    "right away",
    "within",
    "today",
    "tomorrow",
    "bientot",
    "rapidement",
    "sous peu",
    "aujourd'hui",
    "demain",
    "dans les",
)


class MessageKey(StrEnum):
    """One thing a customer can be told. Fewer of these than there are codes."""

    ASK_FOR_ORDER_NUMBER = "ask_for_order_number"
    ASK_WHICH_PRODUCT = "ask_which_product"
    ASK_WHAT_IS_MEANT = "ask_what_is_meant"
    ASK_WHICH_FIRST = "ask_which_first"
    HANDED_TO_A_SPECIALIST = "handed_to_a_specialist"
    ACCOUNT_NOT_LINKED = "account_not_linked"
    BEING_CHECKED_HERE = "being_checked_here"


# Where a request went decides most of what its sender hears, because that is
# what happened to them. A reason refines it, and only where one code calls
# for genuinely different words from another on the same route.
_DEFAULT: dict[Route, MessageKey] = {
    Route.CLARIFICATION: MessageKey.ASK_WHAT_IS_MEANT,
    Route.HUMAN_ESCALATION: MessageKey.HANDED_TO_A_SPECIALIST,
    Route.INTERNAL_REVIEW: MessageKey.BEING_CHECKED_HERE,
}

# Only the codes that change the wording. Which rating fell short, or which
# adapter could not be reached, is ours to know: it reaches the queue entry
# and gives a customer nothing to act on, so it takes the route's own words.
_SAYS: dict[ReasonCode, MessageKey] = {
    ClarificationReason.MISSING_ORDER_ID: MessageKey.ASK_FOR_ORDER_NUMBER,
    ClarificationReason.MISSING_PRODUCT_REFERENCE: MessageKey.ASK_WHICH_PRODUCT,
    ClarificationReason.MULTIPLE_INTENTS: MessageKey.ASK_WHICH_FIRST,
    ClarificationReason.UNRESOLVED_INTENT: MessageKey.ASK_WHAT_IS_MEANT,
    BlockedReason.CUSTOMER_NOT_LINKED: MessageKey.ACCOUNT_NOT_LINKED,
}

# One request can carry several codes and its sender gets one sentence. The
# order is declared rather than derived, so the same set always produces the
# same reply and nobody has to read a sort key to predict it. Stitching two
# sentences together was the alternative, and it reads as a form letter.
_FIRST: tuple[MessageKey, ...] = (
    MessageKey.HANDED_TO_A_SPECIALIST,
    MessageKey.ACCOUNT_NOT_LINKED,
    MessageKey.BEING_CHECKED_HERE,
    MessageKey.ASK_WHICH_FIRST,
    MessageKey.ASK_FOR_ORDER_NUMBER,
    MessageKey.ASK_WHICH_PRODUCT,
    MessageKey.ASK_WHAT_IS_MEANT,
)


class OutcomeMessageError(RuntimeError):
    """The wording on disk cannot be trusted, so none of it loads."""


class NothingToTellThemError(RuntimeError):
    """A request would reach a customer with no way to say what happened."""


class OutcomeMessage(BaseModel):
    """One approved sentence about what became of a request."""

    model_config = _STRICT

    key: MessageKey = Field(strict=False)
    locale: Locale
    version: PositiveInt
    approved: bool
    # Trimmed before it is measured. A length of one is satisfied by a space,
    # which renders as the silence this whole file exists to prevent.
    sentence: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    def model_post_init(self, _: object) -> None:
        if re.search(r"\d", self.sentence):
            raise ValueError("sentence states a figure; nothing here has evidence")
        if _PLACEHOLDER.search(self.sentence):
            raise ValueError("sentence has a slot; these are filled from nothing")
        # Folded, not merely lowercased: the list is written without accents
        # and "bientot" does not appear in "bientôt", so an ordinary French
        # promise went straight through.
        folded = fold(self.sentence)
        promised = [word for word in _PROMISES if fold(word) in folded]
        if promised:
            raise ValueError(f"sentence promises {promised}, which nobody has checked")

    @property
    def reference(self) -> str:
        return f"say:{self.key}.{self.locale}.v{self.version}"

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    @property
    def cited(self) -> str:
        """Reference and digest together, as an audit record keeps them."""
        return f"{self.reference}@{self.content_hash}"


class MessageBook:
    """Every approved sentence, by what it says and in what language."""

    def __init__(self, messages: Sequence[OutcomeMessage]) -> None:
        self._approved: dict[tuple[MessageKey, str], OutcomeMessage] = {}
        for message in messages:
            if not message.approved:
                continue
            at = (message.key, message.locale)
            if at in self._approved:
                raise OutcomeMessageError(
                    f"two approved ways to say {message.key} in {message.locale}"
                )
            self._approved[at] = message

    def __len__(self) -> int:
        return len(self._approved)

    def __iter__(self) -> Iterator[OutcomeMessage]:
        return iter(self._approved.values())

    def tell(
        self, reasons: Sequence[ReasonCode], locale: Locale, route: Route
    ) -> OutcomeMessage:
        """The sentence these reasons come to on this route, in this language.

        The route is not decoration. A request held back here and one passed
        to a specialist can arrive carrying the same code — coverage fell
        short in both — and telling somebody their message went to a colleague
        when it did not is a false statement about what happened to it.
        """
        if not reasons:
            raise NothingToTellThemError("nothing happened that anybody can be told")
        keys = {_SAYS.get(reason, _DEFAULT[route]) for reason in reasons}
        chosen = next((key for key in _FIRST if key in keys), None)
        if chosen is None:
            raise NothingToTellThemError(f"nothing is written for {list(reasons)}")
        message = self._approved.get((chosen, locale))
        if message is None:
            raise NothingToTellThemError(f"nothing says {chosen} in {locale}")
        return message


def load_messages(directory: Path = DEFAULT_MESSAGE_DIR) -> MessageBook:
    """Parse every TOML file in `directory`, or raise without loading any."""
    messages = []
    for path in sorted(directory.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                message = OutcomeMessage.model_validate(tomllib.load(handle))
        except (tomllib.TOMLDecodeError, ValidationError) as exc:
            raise OutcomeMessageError(f"{path.name} is not valid wording") from exc
        expected = f"{message.key}.{message.locale}.v{message.version}.toml"
        if path.name != expected:
            raise OutcomeMessageError(f"{path.name} declares itself {expected}")
        messages.append(message)
    if not messages:
        raise OutcomeMessageError(f"no outcome messages found in {directory}")
    book = MessageBook(messages)
    missing = [
        f"{key} in {locale}"
        for key in MessageKey
        for locale in get_args(Locale)
        if (key, locale) not in book._approved
    ]
    if missing:
        # Checked here rather than only in a test, because a file left out of
        # an image is not a mistake the test suite is present to notice. The
        # process refuses to start instead of answering until it meets the
        # customer whose language went missing.
        raise OutcomeMessageError(f"nothing approved says {missing}")
    return book
