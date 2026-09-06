"""Reading a message with a hosted model, and what its scores are taken to mean."""

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.agent.classification import _INTENTS, _RISKS, HuggingFaceClassifier
from app.agent.intent import (
    ClassifierMisconfiguredError,
    ClassifierUnavailableError,
    Intent,
)
from app.agent.reasons import RiskReason
from app.core.config import Settings

FRAUD = next(h for h, r in _RISKS["en"].items() if r is RiskReason.SUSPECTED_FRAUD)
COMPROMISE = next(
    h for h, r in _RISKS["en"].items() if r is RiskReason.ACCOUNT_COMPROMISE
)
ORDERS = next(h for h, i in _INTENTS["en"].items() if i is Intent.ORDER_STATUS)
RETURNS = next(h for h, i in _INTENTS["en"].items() if i is Intent.RETURN_POLICY)


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, HUGGINGFACE_API_TOKEN=SecretStr("hf_test")
    )


def answering(scores: dict[str, float]) -> tuple[HuggingFaceClassifier, list[Any]]:
    """A classifier whose model returns exactly these scores, plus zero elsewhere."""
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        labels = payload["parameters"]["candidate_labels"]
        return httpx.Response(
            200,
            json={
                "sequence": payload["inputs"],
                "labels": labels,
                "scores": [scores.get(label, 0.01) for label in labels],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HuggingFaceClassifier(settings(), client), sent


def refusing(status: int) -> HuggingFaceClassifier:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "no"})

    return HuggingFaceClassifier(
        settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def returning(body: Any) -> HuggingFaceClassifier:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return HuggingFaceClassifier(
        settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.asyncio
async def test_a_confident_intent_does_not_quiet_a_report_beside_it() -> None:
    """The reason the request is multi-label.

    Labels that share one total take from each other, so a message plainly
    about an order would push every danger score down for no reason except
    that the order score went up. Here both are high and both are heard.
    """
    classifier, _ = answering({ORDERS: 0.95, FRAUD: 0.80})
    read = await classifier.classify("where is the order I never placed", "en")
    assert read.risks == {RiskReason.SUSPECTED_FRAUD}
    assert read.intent is Intent.ORDER_STATUS


@pytest.mark.asyncio
async def test_the_request_asks_for_independent_scores() -> None:
    classifier, sent = answering({ORDERS: 0.9})
    await classifier.classify("where is my order", "en")
    assert sent[0]["parameters"]["multi_label"] is True


@pytest.mark.asyncio
async def test_every_report_in_one_message_is_heard() -> None:
    classifier, _ = answering({FRAUD: 0.9, COMPROMISE: 0.75})
    read = await classifier.classify("someone got in and used my card", "en")
    assert read.risks == {RiskReason.SUSPECTED_FRAUD, RiskReason.ACCOUNT_COMPROMISE}


@pytest.mark.asyncio
async def test_a_faint_suggestion_of_trouble_is_not_a_report() -> None:
    classifier, _ = answering({FRAUD: 0.30})
    assert (await classifier.classify("my order is late", "en")).risks == frozenset()


@pytest.mark.asyncio
async def test_an_unconfident_reading_names_no_intent() -> None:
    """Nothing is guessed. The customer is asked instead."""
    classifier, _ = answering({ORDERS: 0.40})
    assert (await classifier.classify("I need some help", "en")).intent is None


@pytest.mark.asyncio
async def test_two_readings_a_hair_apart_name_no_intent() -> None:
    """Both are confident, which means the message supports both.

    Choosing between them on a rounding error picks the sources the request
    is answered from, so it goes back to the customer instead.
    """
    classifier, _ = answering({ORDERS: 0.91, RETURNS: 0.89})
    assert (
        await classifier.classify("about my order and returns", "en")
    ).intent is None


@pytest.mark.asyncio
async def test_a_french_message_is_read_against_french_sentences() -> None:
    """Entailment compares two pieces of text. Mixing languages asks a harder
    question than the one being put, and the model answers the harder one.
    """
    classifier, sent = answering({})
    await classifier.classify("Où est ma commande ?", "fr")
    asked = set(sent[0]["parameters"]["candidate_labels"])
    assert asked == {*_RISKS["fr"], *_INTENTS["fr"]}
    assert not asked & {*_RISKS["en"], *_INTENTS["en"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_a_refused_token_or_a_missing_model_is_a_mistake_not_weather(
    status: int,
) -> None:
    with pytest.raises(ClassifierMisconfiguredError, match="check DORNASHOP"):
        await refusing(status).classify("hello", "en")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_bad_afternoon_is_something_to_wait_out(status: int) -> None:
    with pytest.raises(ClassifierUnavailableError):
        await refusing(status).classify("hello", "en")


@pytest.mark.asyncio
async def test_the_probe_finds_a_typo_before_a_customer_does() -> None:
    """Constructing it proves a token is present, not that anybody accepts it."""
    with pytest.raises(ClassifierMisconfiguredError):
        await refusing(401).probe()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        ["not", "an", "object"],
        {"scores": [0.9]},
        {"labels": [ORDERS], "scores": [0.9, 0.1]},
        {"labels": ["something we never asked"], "scores": [0.9]},
        {"labels": [ORDERS], "scores": [True]},
        {"labels": [ORDERS], "scores": ["0.9"]},
        {"labels": [ORDERS], "scores": [1.4]},
        {"labels": [ORDERS], "scores": [-0.2]},
    ],
    ids=[
        "not an object",
        "no labels",
        "counts disagree",
        "labels we did not ask",
        "a boolean score",
        "a string score",
        "above one",
        "below zero",
    ],
)
async def test_an_answer_that_is_not_scores_is_refused(body: Any) -> None:
    """Every way of being wrong here is quiet.

    A missing label reads as nought, which is a danger nobody reported. A
    score past one clears any threshold. Left unchecked, the failure shows up
    as a message that should have escalated and did not.
    """
    with pytest.raises(ClassifierUnavailableError):
        await returning(body).classify("hello", "en")


@pytest.mark.asyncio
async def test_a_score_that_compares_false_with_everything_is_refused() -> None:
    """NaN loses every comparison, so a threshold silently stops applying."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        labels = payload["parameters"]["candidate_labels"]
        return httpx.Response(
            200,
            content=json.dumps(
                {"labels": labels, "scores": [float("nan")] * len(labels)}
            ).replace("NaN", "NaN"),
            headers={"content-type": "application/json"},
        )

    classifier = HuggingFaceClassifier(
        settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ClassifierUnavailableError):
        await classifier.classify("hello", "en")


def test_every_risk_and_intent_has_a_sentence_in_every_language() -> None:
    """A label with no sentence is a danger the model is never asked about."""
    for locale in ("en", "fr"):
        assert set(_RISKS[locale].values()) == set(RiskReason)
        assert set(_INTENTS[locale].values()) == set(Intent)
