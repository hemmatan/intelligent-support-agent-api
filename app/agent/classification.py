"""Reading a message with a model, after the phrase rules have had their turn.

Zero-shot classification scores whether a message entails a sentence. What it
is given is therefore sentences, not the names of enum members: asking whether
"I was billed twice" entails "SUSPECTED_FRAUD" asks about a token no customer
has ever written, and the score that comes back is about the token.

Danger and intent are asked in one request and decided by two different rules,
because they are two different questions. "Is the customer in trouble" admits
none, one or several answers at once, and a message reporting both a stolen
card and a hacked account reports both. "What are they asking for" admits one,
and picking it wrongly sends the request to the wrong sources.

That difference is why the request is multi-label. Labels that compete divide
one total between them, so a message plainly about an order would push every
danger score down for no reason except that the order score went up. Scored
independently, one has nothing to say about the other.

Nothing here is a reliability rating. These scores choose which sources get
read; how far what comes back may be trusted is decided later, by evidence.
"""

import math
from typing import Any

import httpx

from app.agent.intent import (
    Classification,
    ClassifierMisconfiguredError,
    ClassifierUnavailableError,
    Intent,
)
from app.agent.knowledge import Locale
from app.agent.reasons import RiskReason
from app.core.config import Settings

# Answers that will be identical however many times they are asked for.
_CONFIGURATION_FAILURES = frozenset((401, 403, 404))

_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"

# The labels below are whole hypotheses, so nothing is wrapped around them.
# Left to its default the provider builds "This example is {}.", which turns a
# sentence into gibberish and puts an English frame around a French one.
_BARE = "{}"

# Written in the customer's language rather than translated at the last
# moment: entailment is judged between two pieces of text, and a French
# message against an English hypothesis is a harder question than the one
# being asked.
_RISKS: dict[Locale, dict[str, RiskReason]] = {
    "en": {
        "the customer says they were billed wrongly or billed twice": (
            RiskReason.PAYMENT_DISPUTE
        ),
        "the customer reports a payment or an order they did not authorise": (
            RiskReason.SUSPECTED_FRAUD
        ),
        "the customer says somebody else has got into their account": (
            RiskReason.ACCOUNT_COMPROMISE
        ),
        "the customer mentions a lawyer or threatens to take legal action": (
            RiskReason.LEGAL_THREAT
        ),
    },
    "fr": {
        "le client dit avoir été facturé à tort ou facturé deux fois": (
            RiskReason.PAYMENT_DISPUTE
        ),
        "le client signale un paiement ou une commande qu'il n'a pas autorisé": (
            RiskReason.SUSPECTED_FRAUD
        ),
        "le client dit que quelqu'un d'autre a accédé à son compte": (
            RiskReason.ACCOUNT_COMPROMISE
        ),
        "le client parle d'un avocat ou menace d'engager une action en justice": (
            RiskReason.LEGAL_THREAT
        ),
    },
}

_INTENTS: dict[Locale, dict[str, Intent]] = {
    "en": {
        "the customer is asking what the returns policy allows": (Intent.RETURN_POLICY),
        "the customer is asking how long delivery takes or where we deliver": (
            Intent.SHIPPING_POLICY
        ),
        "the customer is asking where their own order has got to": (
            Intent.ORDER_STATUS
        ),
        "the customer is asking whether a product is in stock": (
            Intent.PRODUCT_AVAILABILITY
        ),
        "the customer is asking where their refund has got to": (Intent.REFUND_STATUS),
    },
    "fr": {
        "le client demande ce que la politique de retour autorise": (
            Intent.RETURN_POLICY
        ),
        "le client demande les délais de livraison ou les pays desservis": (
            Intent.SHIPPING_POLICY
        ),
        "le client demande où en est sa propre commande": (Intent.ORDER_STATUS),
        "le client demande si un article est en stock": (Intent.PRODUCT_AVAILABILITY),
        "le client demande où en est son remboursement": (Intent.REFUND_STATUS),
    },
}


class HuggingFaceClassifier:
    """Calls a hosted zero-shot model, and decides what its scores mean.

    Failures divide by whether trying again could help, the same way the
    embedder's do. The difference is what a failure costs: retrieval without
    meaning is a worse answer, while a safety pass that did not run is no
    answer at all, so this one stops the request rather than degrading it.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if settings.HUGGINGFACE_API_TOKEN is None:
            raise ValueError("HUGGINGFACE_API_TOKEN is not configured")
        self._token = settings.HUGGINGFACE_API_TOKEN.get_secret_value()
        self._url = _ENDPOINT.format(model=settings.CLASSIFIER_MODEL)
        self._timeout = settings.CLASSIFIER_TIMEOUT_SECONDS
        self._risk_threshold = settings.CLASSIFIER_RISK_THRESHOLD
        self._intent_threshold = settings.CLASSIFIER_INTENT_THRESHOLD
        self._intent_margin = settings.CLASSIFIER_INTENT_MARGIN
        self._client = client

    async def probe(self) -> None:
        """Ask one throwaway question, so a typo is found before a customer is.

        Building the adapter proves a token is present, not that anybody will
        accept it. A wrong model name and a revoked key both look exactly like
        a working configuration until the first message arrives.
        """
        await self.classify("hello", "en")

    async def classify(self, message: str, locale: Locale) -> Classification:
        risks = _RISKS[locale]
        intents = _INTENTS[locale]
        scored = await self._scores(message, [*risks, *intents])

        reported = {
            reason
            for hypothesis, reason in risks.items()
            if scored.get(hypothesis, 0.0) >= self._risk_threshold
        }
        return Classification(
            intent=self._best(scored, intents), risks=frozenset(reported)
        )

    def _best(
        self, scored: dict[str, float], intents: dict[str, Intent]
    ) -> Intent | None:
        """The one intent that is both confident and clear of the field.

        Confident alone is not enough. Two readings a hair apart mean the
        message supports both, and choosing between them by a rounding error
        picks the sources a request is answered from. Where that happens the
        customer is asked instead.
        """
        ranked = sorted(
            (
                (scored.get(hypothesis, 0.0), intent)
                for hypothesis, intent in intents.items()
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked:
            return None
        best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else (0.0, None)
        if best[0] < self._intent_threshold:
            return None
        # Equal scores are ambiguous whatever the margin is set to. Left to the
        # comparison alone, a configured margin of zero made the winner
        # whichever hypothesis happened to be declared first.
        if best[0] == runner_up[0]:
            return None
        if best[0] - runner_up[0] < self._intent_margin:
            return None
        return best[1]

    async def _scores(self, message: str, hypotheses: list[str]) -> dict[str, float]:
        payload = {
            "inputs": message,
            "parameters": {
                "candidate_labels": hypotheses,
                "hypothesis_template": _BARE,
                "multi_label": True,
            },
        }
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            if self._client is not None:
                response = await self._client.post(
                    self._url, json=payload, headers=headers, timeout=self._timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._url, json=payload, headers=headers
                    )
            if response.status_code in _CONFIGURATION_FAILURES:
                raise ClassifierMisconfiguredError(
                    f"{self._url} answered {response.status_code}; check "
                    f"DORNASHOP_HUGGINGFACE_API_TOKEN and DORNASHOP_CLASSIFIER_MODEL"
                )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ClassifierUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        return _scores_from(body, expected=hypotheses)


def _scores_from(body: Any, expected: list[str]) -> dict[str, float]:
    """Read the documented answer: one object per label, each with its score.

    Checked here because nothing downstream checks it again, and every way of
    being wrong is quiet. A label that never came back reads as nought, which
    is a danger nobody reported. A score above one clears any threshold it
    meets. A NaN loses every comparison, so a message that should have gone to
    a person goes on instead.
    """
    if not isinstance(body, list):
        raise ClassifierUnavailableError(f"expected a list of scores, got {body!r:.80}")

    paired: dict[str, float] = {}
    for item in body:
        if not isinstance(item, dict):
            raise ClassifierUnavailableError(f"{item!r:.60} is not a scored label")
        label, score = item.get("label"), item.get("score")
        if not isinstance(label, str):
            raise ClassifierUnavailableError(f"{label!r:.60} is not a label")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ClassifierUnavailableError(f"{score!r} is not a score")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ClassifierUnavailableError(
                f"{score!r} is not a score between 0 and 1"
            )
        if label in paired:
            raise ClassifierUnavailableError(f"{label!r:.60} was scored twice")
        paired[label] = float(score)

    if sorted(paired) != sorted(expected):
        raise ClassifierUnavailableError("the labels answered are not the ones asked")
    return paired
