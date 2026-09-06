"""The only sentences a customer is ever sent.

Every word here was written by a person and approved before it shipped. What
the running service contributes is which template to use and which claim goes
in which slot, and both of those are decided by structure rather than by
phrasing. No model writes to a customer, so no model can invent to one.

Connective language is the reason this is worth the trouble. "Your parcel
should arrive shortly" is a prediction about a delivery, and calling it
phrasing does not make it stop being a claim. The way to be sure nothing like
that is asserted is for there to be nowhere for it to come from.

A template answers one fact. A reply is however many of them the question
asked for, in a fixed order, each filled from the entry that was cited for it.
"""

import hashlib
import json
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from app.agent.facts import Fact
from app.agent.knowledge import (
    Locale,
    PolicyClaims,
    PolicyEntry,
    ReturnPolicyClaims,
    ShippingPolicyClaims,
)

DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "responses"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

# Which claim models are able to state each fact, so a template can be checked
# against the fields that could ever fill it rather than at the moment it is
# needed, in front of somebody waiting.
_STATED_BY: dict[Fact, list[type[PolicyClaims]]] = {}
for _claims in (ReturnPolicyClaims, ShippingPolicyClaims):
    for _field, _fact in _claims.STATES.items():
        _STATED_BY.setdefault(_fact, []).append(_claims)


class ResponseTemplateError(RuntimeError):
    """The approved wording on disk cannot be trusted, so none of it loads."""


class NothingApprovedToSayError(RuntimeError):
    """No approved wording exists for what this answer would have to say.

    A gap in the phrase book, not in the evidence. The request waits for
    somebody here rather than being answered in words nobody signed off.
    """


class UnattributedReplyError(ValueError):
    """Wording for a customer that cannot name where it came from."""


@dataclass(frozen=True)
class ApprovedReply:
    """Sentences a person signed off, inseparable from what they were.

    A plain string would have carried the same words with nothing behind
    them, and anywhere a string is accepted, one composed on the spot is
    accepted too. This can only be built with its provenance attached, so
    holding one is evidence rather than a claim about where it came from.
    """

    text: str
    said: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise UnattributedReplyError("an approved reply that says nothing")
        if not self.said:
            raise UnattributedReplyError(f"{self.text!r} names no approved wording")


class ResponseTemplate(BaseModel):
    """One approved sentence, and the claims allowed to fill it."""

    model_config = _STRICT

    # Written in the file as its value; strict mode will not coerce it alone.
    fact: Fact = Field(strict=False)
    locale: Locale
    version: PositiveInt
    approved: bool
    sentence: str = Field(min_length=1)

    # Approved words for each value a named choice can take. A figure prints
    # as itself in any language; "standard_items" does not, and a slot with no
    # wording for it puts an internal token in front of a customer.
    words: Mapping[str, str] = Field(default_factory=dict)

    def model_post_init(self, _: object) -> None:
        if re.search(r"\d", self.sentence):
            raise ValueError("sentence contains a literal digit; name a claim instead")
        named = set(_PLACEHOLDER.findall(self.sentence))
        if not named:
            raise ValueError("sentence names no claim, so nothing supports it")
        # Only the fields mapped to this fact, not every field on a model that
        # happens to state it. Allowing the rest let approved wording for the
        # standard delivery time answer with the express figure: a correct
        # sentence about the wrong thing, filed as the reply to the other
        # question and rated as covering it.
        stating = [
            {field for field, stated in claims.STATES.items() if stated is self.fact}
            for claims in _STATED_BY.get(self.fact, [])
        ]
        unknown = named - set().union(*stating) if stating else named
        if unknown:
            raise ValueError(f"nothing stating {self.fact} declares {sorted(unknown)}")
        # And all of them. A fact carried by two figures is not settled by one:
        # a range quoted by its lower end alone is a shorter delivery promise
        # than the policy makes.
        if not any(fields <= named for fields in stating):
            raise ValueError(
                f"{self.fact} is stated by {sorted(set().union(*stating))}, and "
                f"the sentence names only {sorted(named)}"
            )
        choices = {
            value
            for claims in _STATED_BY.get(self.fact, [])
            for field in named & claims.model_fields.keys()
            for value in get_args(claims.model_fields[field].annotation)
            if isinstance(value, str)
        }
        unsaid = choices - self.words.keys()
        if unsaid:
            raise ValueError(f"no approved words for {sorted(unsaid)}")
        spare = self.words.keys() - choices
        if spare:
            raise ValueError(f"{sorted(spare)} is not a value any claim can take")

    @property
    def reference(self) -> str:
        return f"say:{self.fact}.{self.locale}.v{self.version}"

    @property
    def content_hash(self) -> str:
        """Digest of the sentence and every word that can go into it.

        The reference names a version. It cannot say whether the wording under
        that version is the wording that went out, and re-editing approved text
        in place is exactly the change nobody announces.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def fill(self, values: Mapping[str, object]) -> str:
        """Put the claim values in, and refuse if one is not there.

        Coverage has already established that the cited entry states this
        fact, so a gap here is a corpus fault rather than a thin answer, and
        it is louder than a sentence delivered with a hole in it.
        """
        named = set(_PLACEHOLDER.findall(self.sentence))
        missing = named - values.keys()
        if missing:
            raise NothingApprovedToSayError(
                f"{self.reference} needs {sorted(missing)}, which the evidence "
                f"does not state"
            )

        def spoken(name: str) -> str:
            value = values[name]
            return self.words[value] if isinstance(value, str) else str(value)

        return _PLACEHOLDER.sub(lambda m: spoken(m.group(1)), self.sentence)


class TemplateLibrary:
    """Every approved sentence, indexed by what it answers and in what language."""

    def __init__(self, templates: Sequence[ResponseTemplate]) -> None:
        self._approved: dict[tuple[Fact, str], ResponseTemplate] = {}
        for template in templates:
            if not template.approved:
                continue
            key = (template.fact, template.locale)
            if key in self._approved:
                raise ResponseTemplateError(
                    f"two approved ways to say {template.fact} in {template.locale}"
                )
            self._approved[key] = template

    def __len__(self) -> int:
        return len(self._approved)

    def __iter__(self) -> Iterator[ResponseTemplate]:
        return iter(self._approved.values())

    def say(
        self, facts: frozenset[Fact], entry: PolicyEntry, locale: Locale
    ) -> ApprovedReply:
        """The reply, and the templates it was built from.

        Ordered by the fact vocabulary rather than by whatever the question
        mentioned first, so the same two facts always read the same way round.
        """
        values = entry.claims.values()
        sentences, used = [], []
        for fact in Fact:
            if fact not in facts:
                continue
            template = self._approved.get((fact, locale))
            if template is None:
                raise NothingApprovedToSayError(
                    f"nothing approved says {fact} in {locale}"
                )
            sentences.append(template.fill(values))
            used.append(f"{template.reference}@{template.content_hash}")
        if not sentences:
            raise NothingApprovedToSayError("an answer that says nothing is not one")
        return ApprovedReply(text=" ".join(sentences), said=tuple(used))


def load_templates(directory: Path = DEFAULT_TEMPLATE_DIR) -> TemplateLibrary:
    """Parse every TOML file in `directory`, or raise without loading any."""
    templates = []
    for path in sorted(directory.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                template = ResponseTemplate.model_validate(tomllib.load(handle))
        except (tomllib.TOMLDecodeError, ValidationError) as exc:
            raise ResponseTemplateError(f"{path.name} is not valid wording") from exc
        expected = f"{template.fact}.{template.locale}.v{template.version}.toml"
        if path.name != expected:
            raise ResponseTemplateError(f"{path.name} declares itself {expected}")
        templates.append(template)
    if not templates:
        raise ResponseTemplateError(f"no response templates found in {directory}")
    return TemplateLibrary(templates)
