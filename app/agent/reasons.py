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


class EscalationReason(StrEnum):
    """Why a person is handling this instead of the service.

    Values are stable: they outlive the code that produces them.
    """

    PAYMENT_DISPUTE = "payment_dispute"
    SUSPECTED_FRAUD = "suspected_fraud"
    ACCOUNT_COMPROMISE = "account_compromise"
    LEGAL_THREAT = "legal_threat"

    # Not a report of trouble, but equally not something to ask about: a
    # customer who is not linked to any commerce record cannot supply the
    # link, so asking spends their turn on a question with no useful answer.
    CUSTOMER_NOT_LINKED = "customer_not_linked"


class ClarificationReason(StrEnum):
    """What the customer is being asked for, and why nothing was answered."""

    UNRESOLVED_INTENT = "unresolved_intent"
    MULTIPLE_INTENTS = "multiple_intents"
    MISSING_ORDER_ID = "missing_order_id"
    MISSING_PRODUCT_REFERENCE = "missing_product_reference"


ReasonCode = EscalationReason | ClarificationReason
"""Either kind, for the audit trail and anywhere counting both."""
