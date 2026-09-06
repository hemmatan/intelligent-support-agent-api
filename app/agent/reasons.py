"""Why a request went where it went.

A closed set, because these are logged, counted, and eventually returned to
callers. Free-form strings would make "how often do we escalate for fraud"
a question about spelling.

Codes are added when something emits them. A code nothing can produce is a
promise about behaviour that does not exist.

They are split by destination rather than listed together. One flat set let a
question to the customer be explained by a fraud report, and a fraud report be
explained by a missing order number: shapes with no meaning, kept out only by
nobody having written them. Which set a code belongs to is also the whole
difference between the two destinations, so nothing separate records it and
nothing separate can contradict it.
"""

from enum import StrEnum


class RiskReason(StrEnum):
    """Danger a message reports, whoever noticed it.

    Separate from the rest because this is the only kind a model may raise.
    Reading a sentence is something it can do. Whether a customer is on file,
    or whether retrieval found anything, are facts about our systems that it
    has not been shown and could only be guessing at — and a guess here is one
    that walks straight into an escalation queue.

    Values are stable: they outlive the code that produces them.
    """

    PAYMENT_DISPUTE = "payment_dispute"
    SUSPECTED_FRAUD = "suspected_fraud"
    ACCOUNT_COMPROMISE = "account_compromise"
    LEGAL_THREAT = "legal_threat"


class BlockedReason(StrEnum):
    """Nothing is wrong with the customer; this cannot be served here.

    Established by looking, never by reading. Each is something the service
    checked and found wanting about its own position.
    """

    # Not a report of trouble, but equally not something to ask about: a
    # customer who is not linked to any commerce record cannot supply the
    # link, so asking spends their turn on a question with no useful answer.
    CUSTOMER_NOT_LINKED = "customer_not_linked"

    # Nothing admissible said anything about what was asked. The gap is the
    # finding: a policy question with no supporting entry is not a weak
    # answer to be softened, it is an answer nobody here can give.
    NO_SUPPORTING_EVIDENCE = "no_supporting_evidence"


EscalationReason = RiskReason | BlockedReason
"""Why a person is handling this instead of the service."""


class ClarificationReason(StrEnum):
    """What the customer is being asked for, and why nothing was answered."""

    UNRESOLVED_INTENT = "unresolved_intent"
    MULTIPLE_INTENTS = "multiple_intents"
    MISSING_ORDER_ID = "missing_order_id"
    MISSING_PRODUCT_REFERENCE = "missing_product_reference"


class ReviewReason(StrEnum):
    """Why this is waiting for somebody here rather than going out.

    Distinct from an escalation: nothing is wrong with the request, something
    is wrong with us. The customer is owed an answer we could not assemble,
    not a specialist.
    """

    SOURCE_UNAVAILABLE = "source_unavailable"

    # The safety pass could not be completed. Not a judgement about the
    # message: a judgement about how much is known about it, which is less
    # than the service is willing to answer on.
    SAFETY_CHECK_UNAVAILABLE = "safety_check_unavailable"


class EvidenceReason(StrEnum):
    """Which rating held an answer back, once the gates had all passed.

    A gate names a condition. These name a judgement, and the route beside
    them says how badly it went: the same shortfall sends a request to a
    colleague at one level and to a specialist at the next.
    """

    UNAUTHORITATIVE = "unauthoritative_evidence"
    NOT_COVERED = "evidence_does_not_cover_the_question"
    POORLY_MATCHED = "evidence_poorly_matched"
    STALE = "evidence_stale"


ReasonCode = EscalationReason | ClarificationReason | ReviewReason | EvidenceReason
"""Any kind, for the audit trail and anywhere counting all of them."""
