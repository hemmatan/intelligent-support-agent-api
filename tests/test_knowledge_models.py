"""Policy entries are strictly typed, and identified by their content."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.agent.knowledge import (
    PolicyEntry,
    PolicyKind,
    ReturnPolicyClaims,
    ReturnPolicyEntry,
    ShippingPolicyClaims,
    ShippingPolicyEntry,
)

ENTRIES: TypeAdapter[ReturnPolicyEntry | ShippingPolicyEntry] = TypeAdapter(PolicyEntry)

VALID = {
    "id": "returns.standard",
    "locale": "en",
    "version": 1,
    "approved": True,
    "kind": "return_policy",
    "claims": {
        "return_window_days": 30,
        "eligibility": "standard_items",
        "excluded_categories": ["underwear"],
        "sale_items_follow_standard_window": True,
        "final_sale_returnable": False,
        "proof_of_purchase_required": True,
    },
    "words": {
        "standard_items": "most items",
        "underwear": "underwear",
        "proof_of_purchase_required_true": "returned with a receipt",
        "proof_of_purchase_required_false": "returned without a receipt",
        "sale_items_follow_standard_window_true": "are treated the same way",
        "sale_items_follow_standard_window_false": "may not be sent back",
        "final_sale_returnable_true": "may be sent back too",
        "final_sale_returnable_false": "may not",
    },
    "prose_template": "Return {eligibility} within {return_window_days} days.",
}


def entry(**overrides: object) -> ReturnPolicyEntry | ShippingPolicyEntry:
    return ENTRIES.validate_python(VALID | overrides)


def test_a_valid_entry_parses() -> None:
    parsed = entry()
    assert isinstance(parsed, ReturnPolicyEntry)
    assert parsed.kind is PolicyKind.RETURN_POLICY
    assert parsed.claims.return_window_days == 30


@pytest.mark.parametrize(
    "claims",
    [
        {"return_window_days": "approximately thirty", "eligibility": "standard_items"},
        {"return_window_days": 0, "eligibility": "standard_items"},
        {"return_window_days": 30, "eligibility": 42},
        {"return_window_days": 30, "eligibility": "whenever_they_like"},
    ],
    ids=["prose-in-a-number", "zero-days", "number-for-an-enum", "unknown-value"],
)
def test_claims_that_are_not_the_declared_type_are_rejected(
    claims: dict[str, object],
) -> None:
    """A dict named "claims" would accept every one of these."""
    with pytest.raises(ValidationError):
        entry(claims=claims)


def test_unknown_claim_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        entry(
            claims={
                "return_window_days": 30,
                "eligibility": "standard_items",
                "restocking_fee": 5,
            }
        )


def test_a_kind_cannot_carry_another_kinds_claims() -> None:
    with pytest.raises(ValidationError):
        entry(kind="shipping_policy")


def test_the_reference_names_the_entry_not_its_content() -> None:
    assert entry().reference == "kb:returns.standard.en.v1"
    assert (
        entry(prose_template="Rewritten, still {return_window_days} days.").reference
        == "kb:returns.standard.en.v1"
    )


def test_the_hash_is_stable_across_field_order() -> None:
    reordered = {k: VALID[k] for k in reversed(list(VALID))}
    assert ENTRIES.validate_python(reordered).content_hash == entry().content_hash


@pytest.mark.parametrize(
    "change",
    [
        {"prose_template": "Send {eligibility} back within {return_window_days} days."},
        {
            "claims": {
                "return_window_days": 14,
                "eligibility": "standard_items",
                "excluded_categories": ["underwear"],
                "sale_items_follow_standard_window": True,
                "final_sale_returnable": False,
                "proof_of_purchase_required": True,
            }
        },
        {"approved": False},
        {"version": 2},
    ],
    ids=["prose", "claims", "approval", "version"],
)
def test_the_hash_changes_when_content_changes(change: dict[str, object]) -> None:
    assert entry(**change).content_hash != entry().content_hash


def test_malformed_identifiers_are_rejected() -> None:
    for bad in ["Returns.Standard", "returns", "returns..standard", ""]:
        with pytest.raises(ValidationError):
            entry(id=bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", "1"), ("approved", "yes")],
    ids=["string-version", "string-boolean"],
)
def test_values_of_the_wrong_type_are_not_quietly_converted(
    field: str, value: str
) -> None:
    """extra="forbid" stops unknown fields; only strict stops coercion."""
    with pytest.raises(ValidationError):
        entry(**{field: value})


def test_a_number_written_as_a_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        entry(claims={"return_window_days": "30", "eligibility": "standard_items"})


def test_a_delivery_range_must_run_forwards() -> None:
    from app.agent.knowledge import ShippingPolicyClaims

    with pytest.raises(ValidationError, match="greater than"):
        ShippingPolicyClaims(
            standard_delivery_days_min=10,
            standard_delivery_days_max=2,
            express_delivery_days=1,
        )
    with pytest.raises(ValidationError, match="slower than standard"):
        ShippingPolicyClaims(
            standard_delivery_days_min=2,
            standard_delivery_days_max=3,
            express_delivery_days=5,
        )


def test_every_figure_declares_the_question_it_answers() -> None:
    """A field missing from STATES is a figure that settles nothing.

    An entry carrying it would look identical to one that could not answer,
    and coverage would rate it on the fields somebody remembered to map.
    """
    for claims in (ReturnPolicyClaims, ShippingPolicyClaims):
        assert set(claims.STATES) == set(claims.model_fields), claims.__name__
