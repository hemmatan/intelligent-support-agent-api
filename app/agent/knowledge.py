"""Approved policy entries: the knowledge the agent is allowed to assert.

Every entry holds the same fact twice. `prose` is what retrieval searches;
`claims` is what fills template slots. Nothing ever reads a value out of a
sentence, which is what keeps grounding a structural check rather than an
attempt to detect falsehood in text.
"""

import hashlib
import json
import tomllib
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)

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


PolicyEntryAdapter: TypeAdapter[ReturnPolicyEntry | ShippingPolicyEntry] = TypeAdapter(
    PolicyEntry
)

DEFAULT_POLICY_DIR = Path(__file__).parent / "policies"


class PolicyCorpusError(RuntimeError):
    """The corpus on disk cannot be trusted, so it is not loaded at all."""


class PolicyCorpus:
    """Every policy entry the agent may draw on.

    Identity is id, locale and version together, so an older version can stay
    in the corpus and keep an audit reference resolvable. Only one version per
    id and locale may be approved at a time — two would leave the choice of
    which policy is in force to whatever happened to sort first.
    """

    def __init__(self, entries: Sequence[ReturnPolicyEntry | ShippingPolicyEntry]):
        self._entries = tuple(entries)
        self._reject_repeated_identities()
        self._reject_competing_approved_versions()

    def _reject_repeated_identities(self) -> None:
        seen: set[tuple[str, str, int]] = set()
        for entry in self._entries:
            identity = (entry.id, entry.locale, entry.version)
            if identity in seen:
                raise PolicyCorpusError(f"{entry.reference} is defined more than once")
            seen.add(identity)

    def _reject_competing_approved_versions(self) -> None:
        approved: dict[tuple[str, str], int] = {}
        for entry in self._entries:
            if not entry.approved:
                continue
            key = (entry.id, entry.locale)
            if key in approved:
                raise PolicyCorpusError(
                    f"{entry.id} ({entry.locale}) has two approved versions: "
                    f"v{approved[key]} and v{entry.version}"
                )
            approved[key] = entry.version

    def __iter__(self) -> Iterator[ReturnPolicyEntry | ShippingPolicyEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def approved(self) -> tuple[ReturnPolicyEntry | ShippingPolicyEntry, ...]:
        """Entries a response may cite. Drafts are readable but never citable."""
        return tuple(entry for entry in self._entries if entry.approved)


def load_corpus(directory: Path = DEFAULT_POLICY_DIR) -> PolicyCorpus:
    """Parse every TOML file in `directory`, or raise without loading any.

    The filename has to restate the identity inside the file. It is redundant
    on purpose: a copied file whose fields were half-edited is the easiest
    mistake to make here and the hardest to notice by reading.
    """
    entries = []
    for path in sorted(directory.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                entry = PolicyEntryAdapter.validate_python(tomllib.load(handle))
        except (tomllib.TOMLDecodeError, ValidationError) as exc:
            raise PolicyCorpusError(f"{path.name} is not a valid policy") from exc
        expected = f"{entry.id}.{entry.locale}.v{entry.version}.toml"
        if path.name != expected:
            raise PolicyCorpusError(f"{path.name} declares itself to be {expected}")
        entries.append(entry)
    if not entries:
        raise PolicyCorpusError(f"no policy files found in {directory}")
    return PolicyCorpus(entries)
