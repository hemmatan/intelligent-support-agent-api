"""One object holding everything an answer is assembled from.

Built once when the process starts and asked once per request. The endpoint
does not gather a corpus, an index, a phrase book and a set of outcome
messages for every customer who writes in; it hands over a question and takes
back a decision.

Keeping the assembly here also keeps the ordering here. Triage runs before
anything is fetched, and gathering runs only for a request triage placed, so
there is no arrangement of these steps for a route handler to get wrong.
"""

from dataclasses import dataclass

from app.agent.answering import Outcome, Sources, plan_for
from app.agent.enquiry import Enquiry
from app.agent.intent import IntentClassifier
from app.agent.messages import MessageBook
from app.agent.responses import TemplateLibrary
from app.agent.triage import Clarify, Escalate, Proceed, triage
from app.agent.triage import Review as TriageReview
from app.schemas.support import SupportReply, replied
from app.services.cases import CaseRecorder, new_reference


@dataclass(frozen=True)
class SupportAgent:
    """The sources, the approved wording, and the way from one to the other."""

    sources: Sources
    templates: TemplateLibrary
    messages: MessageBook
    # Nothing implements this. It is held here so the one place that would
    # pass one to triage already does, and so the path a failing model takes
    # can be exercised without a model existing.
    classifier: IntentClassifier | None = None

    async def answer(
        self, enquiry: Enquiry, *, cases: CaseRecorder, customer: int
    ) -> SupportReply:
        """Decide what to do with a question, write it down, then answer.

        The record is written before the reply is returned, and a failure to
        write it is a failure to reply. Telling somebody their message has
        reached a colleague, with nothing put anywhere a colleague looks, is
        not a routing decision; it is an untrue sentence.

        The language comes off the enquiry rather than being passed in beside
        it, so what a customer is told and what they asked cannot end up in
        different languages.
        """
        reference = new_reference()
        decided = await triage(enquiry, classifier=self.classifier)
        reached: Outcome | Escalate | Clarify | TriageReview = (
            await plan_for(decided, sources=self.sources, templates=self.templates)
            if isinstance(decided, Proceed)
            else decided
        )
        reply = replied(
            reached,
            messages=self.messages,
            locale=enquiry.locale,
            case=reference,
        )
        await cases.record(reference, enquiry, reply, customer)
        return reply
