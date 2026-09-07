"""Fixed end-to-end decisions for the support-agent pipeline.

The mandatory-escalation assertion is 100% recall over this versioned table,
not an estimate of recall on real customer language. A fixed suite can prevent
known cases from regressing; it cannot show how many unknown phrasings the
deterministic vocabulary will recognise.
"""

import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.answering import Handover, Plan, Sources, plan_for
from app.agent.demo import DemoStorefront
from app.agent.enquiry import Enquiry
from app.agent.facts import Fact, facts_in
from app.agent.intent import Intent, intents_in
from app.agent.knowledge import Locale, load_corpus
from app.agent.profiles import Source
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    EvidenceReason,
    ReasonCode,
    ReviewReason,
    RiskReason,
)
from app.agent.reliability import Route
from app.agent.responses import load_templates
from app.agent.retrieval import PolicyIndex
from app.agent.risk import risks_in
from app.agent.triage import Clarify, Escalate, Proceed, triage
from app.agent.triage import Review as TriageReview

SCENARIO_FILE = Path(__file__).with_name("scenarios.toml")
CUSTOMER = 1


class SuiteMetadata(BaseModel):
    """The fixed clock and the deliberately narrow evaluation claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    observed_at: datetime
    scope: str

    @model_validator(mode="after")
    def require_an_aware_clock(self) -> Self:
        if self.observed_at.tzinfo is None:
            raise ValueError("the scenario clock must include a UTC offset")
        return self


class Scenario(BaseModel):
    """One message, its context, and every decision it is expected to produce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    message: str
    locale: Locale
    account_linked: bool
    order_number: str | None = None
    product_reference: str | None = None
    evidence_age_minutes: int = Field(default=0, ge=0)
    mandatory_escalation: bool
    risk_subject: RiskReason | None = None
    risk_form: Literal["report", "enquiry"] | None = None
    tags: frozenset[str]

    expected_intents: frozenset[Intent]
    expected_risks: frozenset[RiskReason]
    expected_sources: frozenset[Source]
    expected_claims: frozenset[Fact]
    expected_route: Route
    expected_reasons: frozenset[ReasonCode]

    @model_validator(mode="after")
    def require_a_complete_risk_boundary(self) -> Self:
        if (self.risk_subject is None) != (self.risk_form is None):
            raise ValueError("risk_subject and risk_form must be declared together")
        return self


class ScenarioTable(BaseModel):
    """Strictly validated so a misspelled expectation cannot disappear."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite: SuiteMetadata
    scenario: tuple[Scenario, ...]

    @model_validator(mode="after")
    def require_unique_ids(self) -> Self:
        ids = [case.id for case in self.scenario]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return self


TABLE = ScenarioTable.model_validate(tomllib.loads(SCENARIO_FILE.read_text()))


@dataclass(frozen=True)
class Result:
    """The observable decision made for one scenario."""

    scenario: Scenario
    intents: frozenset[Intent]
    risks: frozenset[RiskReason]
    sources: frozenset[Source]
    claims: frozenset[Fact]
    route: Route
    reasons: frozenset[ReasonCode]
    citation_count: int
    answer_wording: str | None


def _route_and_reasons(
    outcome: Plan | Handover | Escalate | Clarify | TriageReview,
) -> tuple[Route, frozenset[ReasonCode]]:
    """Normalize the typed outcomes without weakening their distinctions."""
    if isinstance(outcome, Escalate | Handover):
        return Route.HUMAN_ESCALATION, frozenset(outcome.reasons)
    if isinstance(outcome, Clarify):
        return Route.CLARIFICATION, frozenset({outcome.reason})
    if isinstance(outcome, TriageReview):
        return Route.INTERNAL_REVIEW, frozenset({outcome.reason})

    # A plan by elimination, so the field is named rather than looked up.
    # Fetched through getattr with a default, a rename would go on answering
    # None and this would keep reporting a destination with no cause behind
    # it — silently, and in the one file whose job is noticing drift.
    reasons: frozenset[ReasonCode] = (
        frozenset({outcome.held_back})
        if outcome.held_back is not None
        else outcome.reasons
    )
    return outcome.route, reasons


async def _run(case: Scenario) -> Result:
    observed_at = TABLE.suite.observed_at
    decision_time = observed_at + timedelta(minutes=case.evidence_age_minutes)
    sources = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=DemoStorefront(now=lambda: observed_at),
        now=lambda: decision_time,
    )
    enquiry = Enquiry(
        message=case.message,
        locale=case.locale,
        customer=CUSTOMER if case.account_linked else None,
        order=case.order_number,
        product=case.product_reference,
    )

    # Explicitly no model: this is the deterministic configuration that ships.
    triaged = await triage(enquiry, classifier=None)
    source_plan = triaged.sources if isinstance(triaged, Proceed) else frozenset()
    claims = facts_in(case.message) if isinstance(triaged, Proceed) else frozenset()
    outcome = (
        await plan_for(triaged, sources=sources, templates=load_templates())
        if isinstance(triaged, Proceed)
        else triaged
    )
    route, reasons = _route_and_reasons(outcome)
    citations = outcome.citations if isinstance(outcome, Plan) else ()
    wording = (
        outcome.reply.text if isinstance(outcome, Plan) and outcome.reply else None
    )
    return Result(
        scenario=case,
        intents=intents_in(case.message),
        risks=risks_in(case.message),
        sources=source_plan,
        claims=claims,
        route=route,
        reasons=reasons,
        citation_count=len(citations),
        answer_wording=wording,
    )


async def _run_suite() -> tuple[Result, ...]:
    results = []
    for case in TABLE.scenario:
        results.append(await _run(case))
    return tuple(results)


def _differences(result: Result) -> dict[str, object]:
    case = result.scenario
    compared = {
        "intents": (case.expected_intents, result.intents),
        "risks": (case.expected_risks, result.risks),
        "sources": (case.expected_sources, result.sources),
        "claims": (case.expected_claims, result.claims),
        "route": (case.expected_route, result.route),
        "reasons": (case.expected_reasons, result.reasons),
    }
    return {
        field: {"expected": expected, "actual": actual}
        for field, (expected, actual) in compared.items()
        if expected != actual
    }


def test_the_table_covers_the_specified_scenario_families() -> None:
    """Guard the matrix itself, not only the behavior of the rows it contains."""
    assert "not real-world recall" in TABLE.suite.scope

    boundaries = {
        (case.risk_subject, case.risk_form, case.locale)
        for case in TABLE.scenario
        if case.risk_subject is not None
    }
    required_boundaries = {
        (risk, form, locale)
        for risk in RiskReason
        for form in ("report", "enquiry")
        for locale in ("en", "fr")
    }
    assert required_boundaries <= boundaries
    assert [
        case.id
        for case in TABLE.scenario
        if case.risk_form is not None
        and case.mandatory_escalation is not (case.risk_form == "report")
    ] == []

    declared_intents: set[Intent] = set().union(
        *(case.expected_intents for case in TABLE.scenario)
    )
    assert declared_intents == set(Intent)

    tags: set[str] = set().union(*(case.tags for case in TABLE.scenario))
    assert {
        "unresolved",
        "multiple",
        "policy_full",
        "policy_partial",
        "policy_unsupported",
        "french_delivery",
        "missing_order",
        "unlinked_account",
        "commerce_found",
        "commerce_not_found",
        "commerce_other_customer",
        "commerce_stale",
    } <= tags


@pytest.mark.asyncio
async def test_fixed_scenarios_match_their_declared_decisions() -> None:
    results = await _run_suite()
    disagreements = {
        result.scenario.id: difference
        for result in results
        if (difference := _differences(result))
    }
    assert disagreements == {}


@pytest.mark.asyncio
async def test_invariants_hold_over_the_whole_fixed_suite() -> None:
    """These are table-wide safety properties, not row-by-row examples."""
    results = await _run_suite()

    mandatory = [result for result in results if result.scenario.mandatory_escalation]
    assert mandatory
    assert [
        result.scenario.id
        for result in mandatory
        if result.route is not Route.HUMAN_ESCALATION
    ] == []

    direct = [result for result in results if result.route is Route.DIRECT_RESPONSE]
    assert direct
    assert [result.scenario.id for result in direct if result.citation_count == 0] == []

    assert [
        result.scenario.id
        for result in results
        if result.route is not Route.DIRECT_RESPONSE
        and result.answer_wording is not None
    ] == []

    allowed_reason_types: dict[Route, tuple[type[object], ...]] = {
        Route.DIRECT_RESPONSE: (),
        Route.CLARIFICATION: (ClarificationReason,),
        Route.INTERNAL_REVIEW: (ReviewReason, EvidenceReason),
        Route.HUMAN_ESCALATION: (RiskReason, BlockedReason, EvidenceReason),
    }
    illegal = {
        result.scenario.id: sorted(str(reason) for reason in result.reasons)
        for result in results
        if any(
            not isinstance(reason, allowed_reason_types[result.route])
            for reason in result.reasons
        )
        or (result.route is Route.DIRECT_RESPONSE and result.reasons)
    }
    assert illegal == {}
    assert [
        result.scenario.id
        for result in results
        if result.route is not Route.DIRECT_RESPONSE and not result.reasons
    ] == []
