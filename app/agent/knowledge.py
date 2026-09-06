"""Approved policy entries: the knowledge the agent is allowed to assert.

A figure is authored once, in `claims`. The searchable sentences are written
as `prose_template` with the figure named rather than repeated, and rendered
when the corpus loads. Nothing reads a value out of a sentence, and no value
is written down twice, so the two cannot come to disagree.

The templates here produce text for retrieval to search. Customer-visible
text comes from response templates, which are a different thing entirely.
"""

import hashlib
import json
import re
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from app.agent.facts import Fact

Locale = Literal["en", "fr"]

# Only a bare name. Attribute access, indexing and format specifications are
# not forbidden so much as inexpressible.
_PLACEHOLDER = re.compile(r"\{(\w+)\}")

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class PolicyKind(StrEnum):
    """Selects which claims a policy is allowed to make."""

    RETURN_POLICY = "return_policy"
    SHIPPING_POLICY = "shipping_policy"


class PolicyClaims(BaseModel):
    """Figures a policy states, and the questions each of them answers.

    STATES pairs every field with what asking for it would be asking for, so
    what an entry can settle is read off the same declaration that holds the
    values. A corpus test refuses a field missing from it: a figure nobody
    mapped is a figure that silently answers nothing, and an entry carrying it
    would look no different from one that could not.
    """

    model_config = _STRICT

    STATES: ClassVar[Mapping[str, Fact]] = {}

    @property
    def facts(self) -> frozenset[Fact]:
        """What this entry is able to settle."""
        return frozenset(self.STATES.values())

    def values(self) -> dict[str, object]:
        """Every claim as it was authored, with nothing turned into prose yet.

        A named choice leaves here as the token it is stored as. Which words
        stand for it is customer-facing text, so it lives in the approved
        wording with the rest of the customer-facing text, where it is
        versioned and hashed alongside the sentence that uses it.
        """
        return dict(self.model_dump())


class ReturnPolicyClaims(PolicyClaims):
    """What a return policy may assert."""

    return_window_days: PositiveInt
    eligibility: Literal["standard_items", "all_items", "selected_items"]

    STATES: ClassVar[Mapping[str, Fact]] = {
        "return_window_days": Fact.RETURN_WINDOW,
        "eligibility": Fact.RETURN_ELIGIBILITY,
    }


class ShippingPolicyClaims(PolicyClaims):
    """What a shipping policy may assert."""

    STATES: ClassVar[Mapping[str, Fact]] = {
        "standard_delivery_days_min": Fact.STANDARD_DELIVERY_TIME,
        "standard_delivery_days_max": Fact.STANDARD_DELIVERY_TIME,
        "express_delivery_days": Fact.EXPRESS_DELIVERY_TIME,
    }

    standard_delivery_days_min: PositiveInt
    standard_delivery_days_max: PositiveInt
    express_delivery_days: PositiveInt

    @model_validator(mode="after")
    def check_the_range_runs_forwards(self) -> "ShippingPolicyClaims":
        if self.standard_delivery_days_min > self.standard_delivery_days_max:
            raise ValueError(
                "standard_delivery_days_min is greater than standard_delivery_days_max"
            )
        if self.express_delivery_days > self.standard_delivery_days_min:
            raise ValueError("express delivery is slower than standard delivery")
        return self


class BasePolicyEntry(BaseModel):
    """Identity, approval state, and the two representations of the policy."""

    model_config = _STRICT

    id: str = Field(pattern=r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+$")
    locale: Locale
    version: PositiveInt
    approved: bool
    claims: PolicyClaims
    prose_template: str = Field(min_length=1)

    @field_validator("prose_template")
    @classmethod
    def strip_surrounding_whitespace(cls, value: str) -> str:
        """TOML multiline strings open on a newline; that is not authored text."""
        return value.strip()

    @model_validator(mode="after")
    def check_every_figure_comes_from_a_claim(self) -> "BasePolicyEntry":
        """Refuse a template that could disagree with its claims.

        Digits are banned outright, which is a real constraint: any number
        worth stating in policy prose has to become a structured claim, even
        ones that are not really policy facts. A dispatch cutoff of 3pm needs
        a claim or it needs deleting. That price buys an entry where every
        figure in the text has exactly one authored source.
        """
        if re.search(r"\d", self.prose_template):
            raise ValueError(
                "prose_template contains a literal digit; name a claim instead"
            )
        declared = self.claims.model_dump()
        used = set(_PLACEHOLDER.findall(self.prose_template))
        unknown = used - declared.keys()
        if unknown:
            raise ValueError(f"prose_template names undeclared {sorted(unknown)}")
        numeric = {
            name
            for name, value in declared.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        unused = numeric - used
        if unused:
            raise ValueError(f"prose_template never states {sorted(unused)}")
        return self

    @property
    def prose(self) -> str:
        """The sentences retrieval searches, rendered from the claims."""
        declared = self.claims.model_dump()
        return _PLACEHOLDER.sub(
            lambda m: str(declared[m.group(1)]), self.prose_template
        )

    @property
    def facts(self) -> frozenset[Fact]:
        """What this entry can settle, whatever question brought it back."""
        return self.claims.facts

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
        self._reject_approved_versions_that_disagree_across_locales()
        self._reject_claims_that_differ_by_locale()

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

    def _reject_approved_versions_that_disagree_across_locales(self) -> None:
        """Every language has to be showing the same version of a policy.

        The per-locale check above allows one approved version each, and the
        claim comparison below only runs on entries sharing a version. Between
        them, an approved English v2 and an approved French v1 pass unnoticed
        and quote different rules to different customers.
        """
        versions: dict[str, dict[str, int]] = defaultdict(dict)
        for entry in self._entries:
            if entry.approved:
                versions[entry.id][entry.locale] = entry.version
        for policy_id, by_locale in versions.items():
            if len(set(by_locale.values())) > 1:
                stated = ", ".join(
                    f"{locale} v{version}"
                    for locale, version in sorted(by_locale.items())
                )
                raise PolicyCorpusError(f"{policy_id} is approved at {stated}")

    def _reject_claims_that_differ_by_locale(self) -> None:
        """One policy at one version means one rule, whatever language it is in.

        Translations differ; the business rule does not. A jurisdiction with a
        genuinely different rule is a different policy id, not a translation of
        this one.
        """
        grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
        for entry in self._entries:
            grouped[(entry.id, entry.version)][entry.locale] = entry.claims.model_dump()
        for (policy_id, version), by_locale in grouped.items():
            if len(by_locale) < 2:
                continue
            kinds = {
                entry.kind
                for entry in self._entries
                if (entry.id, entry.version) == (policy_id, version)
            }
            if len(kinds) > 1:
                raise PolicyCorpusError(
                    f"{policy_id} v{version} is a {' and a '.join(sorted(kinds))}"
                )
            fields = {
                name
                for name in next(iter(by_locale.values()))
                if len({str(claims[name]) for claims in by_locale.values()}) > 1
            }
            if fields:
                raise PolicyCorpusError(
                    f"{policy_id} v{version} states {sorted(fields)} differently "
                    f"in {sorted(by_locale)}"
                )

    def __iter__(self) -> Iterator[ReturnPolicyEntry | ShippingPolicyEntry]:
        """Approved entries only.

        Iterating is what callers reach for, so it is the safe operation.
        Reaching a draft takes a method whose name says what it is doing.
        """
        return iter(entry for entry in self._entries if entry.approved)

    def __len__(self) -> int:
        return sum(1 for entry in self._entries if entry.approved)

    def including_drafts(self) -> tuple[ReturnPolicyEntry | ShippingPolicyEntry, ...]:
        """Every entry, approved or not. Nothing citable comes from here."""
        return self._entries


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
