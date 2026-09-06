"""Reading a message with a hosted model, and what its scores are taken to mean."""

import json
import os
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
            json=[
                {"label": label, "score": scores.get(label, 0.01)} for label in labels
            ],
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
async def test_nothing_is_wrapped_around_a_sentence_that_is_already_one() -> None:
    """Left to the default the provider builds "This example is {}."

    Which turns a hypothesis into gibberish, and puts an English frame around
    the French ones — undoing the reason they were written in French.
    """
    classifier, sent = answering({})
    await classifier.classify("où est ma commande", "fr")
    assert sent[0]["parameters"]["hypothesis_template"] == "{}"


@pytest.mark.asyncio
async def test_no_undocumented_keys_are_sent() -> None:
    """An option the provider never documented is one nothing promises to ignore."""
    classifier, sent = answering({})
    await classifier.classify("hello", "en")
    assert "options" not in sent[0]
    assert set(sent[0]) == {"inputs", "parameters"}
    assert set(sent[0]["parameters"]) == {
        "candidate_labels",
        "hypothesis_template",
        "multi_label",
    }


@pytest.mark.asyncio
async def test_the_documented_endpoint_is_the_one_called() -> None:
    """A path the provider does not serve answers 404, which this code reads
    as a model nobody can reach — so a wrong URL reports itself as a wrong
    model name and startup fails describing the wrong thing.
    """
    classifier, _ = answering({})
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json=[])

    classifier = HuggingFaceClassifier(
        settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ClassifierUnavailableError):
        await classifier.classify("hello", "en")
    assert sent == [
        "https://router.huggingface.co/hf-inference/models/"
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    ]


@pytest.mark.asyncio
async def test_an_exact_tie_names_nothing_whatever_the_margin_is() -> None:
    """Configured to nought, the comparison alone let the first-declared win.

    Two identical scores are the clearest case of a message supporting both
    readings, and it was the one case the rule stopped covering.
    """
    tied = Settings(  # type: ignore[call-arg]
        _env_file=None,
        HUGGINGFACE_API_TOKEN=SecretStr("hf_test"),
        CLASSIFIER_INTENT_MARGIN=0.0,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        labels = payload["parameters"]["candidate_labels"]
        scores = {ORDERS: 0.9, RETURNS: 0.9}
        return httpx.Response(
            200,
            json=[{"label": x, "score": scores.get(x, 0.01)} for x in labels],
        )

    classifier = HuggingFaceClassifier(
        tied, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert (await classifier.classify("order and returns", "en")).intent is None


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
        {"labels": [ORDERS], "scores": [0.9]},
        ["not an object"],
        [{"score": 0.9}],
        [{"label": ORDERS}],
        [{"label": ORDERS, "score": 0.9}],
        [{"label": "something we never asked", "score": 0.9}],
        [{"label": ORDERS, "score": True}],
        [{"label": ORDERS, "score": "0.9"}],
        [{"label": ORDERS, "score": 1.4}],
        [{"label": ORDERS, "score": -0.2}],
    ],
    ids=[
        "the shape the old code expected",
        "not an object",
        "no label",
        "no score",
        "only one of the labels asked",
        "a label we did not ask",
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
                [{"label": label, "score": float("nan")} for label in labels]
            ),
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


# Messages a real model has to get right, in both languages a customer writes
# in. Everything above proves the adapter reads scores correctly; only these
# say the scores are worth reading, and only these exercise the URL, the
# payload and the response shape the provider actually serves.
GRADED: list[tuple[str, str, set[RiskReason], Intent | None]] = [
    (
        "en",
        "I was charged twice for the same order",
        {RiskReason.PAYMENT_DISPUTE},
        None,
    ),
    (
        "en",
        "Someone got into my account and ordered things I never bought",
        {RiskReason.ACCOUNT_COMPROMISE, RiskReason.SUSPECTED_FRAUD},
        None,
    ),
    ("en", "How can I keep my account safe from fraud?", set(), None),
    ("en", "How long do I have to send a jacket back?", set(), Intent.RETURN_POLICY),
    ("en", "Where has my parcel got to?", set(), Intent.ORDER_STATUS),
    (
        "fr",
        "J'ai été débité deux fois pour la même commande",
        {RiskReason.PAYMENT_DISPUTE},
        None,
    ),
    ("fr", "Comment protéger mon compte contre la fraude ?", set(), None),
    ("fr", "Quel est le délai de livraison ?", set(), Intent.SHIPPING_POLICY),
]


@pytest.mark.skipif(
    not os.environ.get("DORNASHOP_HUGGINGFACE_API_TOKEN"),
    reason="needs a real token; set DORNASHOP_HUGGINGFACE_API_TOKEN to run",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "message", "risks", "intent"),
    GRADED,
    ids=[f"{locale}: {message[:38]}" for locale, message, _, _ in GRADED],
)
async def test_a_real_model_reads_these_the_way_the_thresholds_assume(
    locale: str, message: str, risks: set[RiskReason], intent: Intent | None
) -> None:
    """The claim every mock in this file deliberately refuses to make.

    A stand-in returns the shape the code expects, so it cannot say whether
    the provider serves that shape, whether the URL is the right one, or
    whether these hypotheses separate a fraud report from a question about
    fraud. Two of those were wrong at once and everything here still passed.

    Skipped by default: it needs a token and the network, and CI has neither.
    """
    read = await HuggingFaceClassifier(Settings()).classify(message, locale)  # type: ignore[arg-type]
    assert read.risks == risks
    assert read.intent is intent


@pytest.mark.skipif(
    not os.environ.get("DORNASHOP_HUGGINGFACE_API_TOKEN"),
    reason="needs a real token; set DORNASHOP_HUGGINGFACE_API_TOKEN to run",
)
@pytest.mark.asyncio
async def test_a_real_provider_accepts_the_request_we_send() -> None:
    """The probe, against the thing it exists to check."""
    await HuggingFaceClassifier(Settings()).probe()
