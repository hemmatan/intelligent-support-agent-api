"""Approved policy entries: the knowledge the agent is allowed to assert.

Every entry holds the same fact twice. `prose` is what retrieval searches;
`claims` is what fills template slots. Nothing ever reads a value out of a
sentence, which is what keeps grounding a structural check rather than an
attempt to detect falsehood in text.
"""

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

Locale = Literal["en", "fr"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class PolicyKind(StrEnum):
    """Selects which claims a policy is allowed to make."""

    RETURN_POLICY = "return_policy"
    SHIPPING_POLICY = "shipping_policy"


class ReturnPolicyClaims(BaseModel):
    """What a return policy may assert."""

    model_config = _STRICT

    return_window_days: PositiveInt
    eligibility: Literal["standard_items", "all_items", "selected_items"]


class ShippingPolicyClaims(BaseModel):
    """What a shipping policy may assert."""

    model_config = _STRICT

    standard_delivery_days_min: PositiveInt
    standard_delivery_days_max: PositiveInt
    express_delivery_days: PositiveInt


class BasePolicyEntry(BaseModel):
    """Identity, approval state, and the two representations of the policy."""

    model_config = _STRICT

    id: str = Field(pattern=r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+$")
    locale: Locale
    version: PositiveInt
    approved: bool
    prose: str = Field(min_length=1)

    @property
    def reference(self) -> str:
        """Identity of this entry, without its content.

        A reference says which entry was used. `content_hash` says which
        version of its content, which is the part that can change while the
        reference stays the same.
        """
        return f"kb:{self.id}.{self.locale}.v{self.version}"

    @property
    def content_hash(self) -> str:
        """Digest of the parsed entry, not of the file that carried it.

        Hashing a canonical dump means reordering fields or editing a comment
        leaves the identity alone, while changing a claim or a word of prose
        does not.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


class ReturnPolicyEntry(BasePolicyEntry):
    kind: Literal[PolicyKind.RETURN_POLICY]
    claims: ReturnPolicyClaims


class ShippingPolicyEntry(BasePolicyEntry):
    kind: Literal[PolicyKind.SHIPPING_POLICY]
    claims: ShippingPolicyClaims


PolicyEntry = Annotated[
    ReturnPolicyEntry | ShippingPolicyEntry, Field(discriminator="kind")
]
