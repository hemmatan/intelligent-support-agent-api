"""Why a request went where it went.

A closed set, because these are logged, counted, and eventually returned to
callers. Free-form strings would make "how often do we escalate for fraud"
a question about spelling.

Codes are added when something emits them. A code nothing can produce is a
promise about behaviour that does not exist.
"""

from enum import StrEnum


class ReasonCode(StrEnum):
    """Values are stable: they outlive the code that produces them."""

    PAYMENT_DISPUTE = "payment_dispute"
    SUSPECTED_FRAUD = "suspected_fraud"
    ACCOUNT_COMPROMISE = "account_compromise"
    LEGAL_THREAT = "legal_threat"

    CUSTOMER_NOT_LINKED = "customer_not_linked"
    MISSING_ORDER_ID = "missing_order_id"
    MISSING_PRODUCT_REFERENCE = "missing_product_reference"
