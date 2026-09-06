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

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from app.agent.knowledge import Locale
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    EvidenceReason,
    ReasonCode,
    ReviewReason,
    RiskReason,
)

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


# Every code that can reach a customer, and what they are told about it.
# Internal detail collapses: which rating fell short, or which source was
# unreachable, is a fact about us and reaches the queue entry, not the reply.
_SAYS: dict[ReasonCode, MessageKey] = {
    ClarificationReason.MISSING_ORDER_ID: MessageKey.ASK_FOR_ORDER_NUMBER,
    ClarificationReason.MISSING_PRODUCT_REFERENCE: MessageKey.ASK_WHICH_PRODUCT,
    ClarificationReason.UNRESOLVED_INTENT: MessageKey.ASK_WHAT_IS_MEANT,
    ClarificationReason.MULTIPLE_INTENTS: MessageKey.ASK_WHICH_FIRST,
    RiskReason.PAYMENT_DISPUTE: MessageKey.HANDED_TO_A_SPECIALIST,
    RiskReason.SUSPECTED_FRAUD: MessageKey.HANDED_TO_A_SPECIALIST,
    RiskReason.ACCOUNT_COMPROMISE: MessageKey.HANDED_TO_A_SPECIALIST,
    RiskReason.LEGAL_THREAT: MessageKey.HANDED_TO_A_SPECIALIST,
    BlockedReason.CUSTOMER_NOT_LINKED: MessageKey.ACCOUNT_NOT_LINKED,
    BlockedReason.NO_SUPPORTING_EVIDENCE: MessageKey.HANDED_TO_A_SPECIALIST,
    EvidenceReason.UNAUTHORITATIVE: MessageKey.HANDED_TO_A_SPECIALIST,
    EvidenceReason.NOT_COVERED: MessageKey.HANDED_TO_A_SPECIALIST,
    EvidenceReason.POORLY_MATCHED: MessageKey.HANDED_TO_A_SPECIALIST,
    EvidenceReason.STALE: MessageKey.HANDED_TO_A_SPECIALIST,
    ReviewReason.SOURCE_UNAVAILABLE: MessageKey.BEING_CHECKED_HERE,
    ReviewReason.NOTHING_APPROVED_TO_SAY: MessageKey.BEING_CHECKED_HERE,
    ReviewReason.INTENT_CHECK_UNAVAILABLE: MessageKey.BEING_CHECKED_HERE,
}

# One request can carry several codes, and a customer gets one sentence. The
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
    sentence: str = Field(min_length=1)

    def model_post_init(self, _: object) -> None:
        if re.search(r"\d", self.sentence):
            raise ValueError("sentence states a figure; nothing here has evidence")
        if _PLACEHOLDER.search(self.sentence):
            raise ValueError("sentence has a slot; these are filled from nothing")
        folded = self.sentence.casefold()
        promised = [word for word in _PROMISES if word in folded]
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

    def tell(self, reasons: Sequence[ReasonCode], locale: Locale) -> OutcomeMessage:
        """The one sentence these reasons come to, in this language."""
        if not reasons:
            raise NothingToTellThemError("nothing happened that anybody can be told")
        keys = {_SAYS[reason] for reason in reasons if reason in _SAYS}
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
    return MessageBook(messages)
