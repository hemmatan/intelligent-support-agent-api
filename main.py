"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.agent.answering import Sources
from app.agent.embedding import Embedder
from app.agent.inference import HuggingFaceEmbedder
from app.agent.knowledge import load_corpus
from app.agent.messages import load_messages
from app.agent.responses import load_templates
from app.agent.retrieval import PolicyIndex
from app.api.auth import router as auth_router
from app.api.cases import router as cases_router
from app.api.health import router as health_router
from app.api.support import router as support_router
from app.core.config import settings
from app.db.session import sessionmanager
from app.services.support import SupportAgent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Function that handles startup and shutdown events.
    To understand more, read https://fastapi.tiangolo.com/advanced/events/

    close() releases the engine permanently, so this application object
    supports a single lifespan cycle per process.
    """
    # Policy that fails validation should stop the process here, while nobody
    # is waiting on an answer, rather than on the first customer to ask.
    corpus = load_corpus()
    app.state.policies = corpus

    index = PolicyIndex(corpus, _embedder())
    # Reaches the network when a token is set. Failing leaves the index
    # lexical, which is a worse service rather than no service.
    await index.warm()
    app.state.policy_index = index

    # Wording is parsed and checked here for the reason the corpus is: a
    # sentence naming a claim nothing states, or a reason nothing can say,
    # should stop the process while nobody is waiting on a reply.
    app.state.support = SupportAgent(
        sources=Sources(knowledge_base=index),
        templates=load_templates(),
        messages=load_messages(),
    )

    yield
    if sessionmanager._engine is not None:
        # Close the DB connection
        await sessionmanager.close()


def _embedder() -> Embedder | None:
    """The hosted embedder when a token is configured, otherwise none."""
    if settings.HUGGINGFACE_API_TOKEN is None:
        return None
    return HuggingFaceEmbedder(settings)


# Rendered at the top of the interactive docs. Markdown, because the four
# outcomes are the first thing anybody reading this API needs to understand
# and a one-line summary cannot carry them.
ASSETS = Path(__file__).parent / "assets"
LOGO = "/assets/dorna-shop-logo.png"
MARK = "/assets/dorna-shop-mark.png"

DESCRIPTION = f"""
<img src="{LOGO}" alt="DornaShop" width="280">

{settings.PROJECT_DESCRIPTION}.

Ask a question at `POST {settings.API_V1_PREFIX}/support/messages` and one of
four things comes back, all of them `200`:

| Route | What happened |
|---|---|
| `direct_response` | Answered from approved wording, with the evidence it rests on |
| `clarification` | Something is missing that you can supply, and you are asked for it |
| `human_escalation` | A person is taking it, and a case is open for them |
| `internal_review` | Nothing is wrong with the request; somebody here finishes it |

Every reply carries a `case` — the record the decision was written into before
the reply was sent — and a `reliability` rating naming each dimension on an
ordinal scale rather than as a percentage.

No sentence a customer receives is written by a model. Figures come from
structured claims and the sentences around them are approved, versioned and
content-hashed.
"""

TAGS = [
    {
        "name": "support",
        "description": "Ask the agent a question and receive one of four decisions.",
    },
    {"name": "authentication", "description": "Accounts, tokens and API keys."},
    {
        "name": "staff",
        "description": "The queue of requests the agent declined to answer.",
    },
    {"name": "system", "description": "Liveness."},
]

app = FastAPI(
    lifespan=lifespan,
    debug=settings.DEBUG,
    title=settings.PROJECT_NAME,
    description=DESCRIPTION,
    openapi_tags=TAGS,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    # Served below instead, so the tab carries the shop's mark rather than
    # the framework's.
    docs_url=None,
    redoc_url=None,
)

app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")


@app.get(f"{settings.API_V1_PREFIX}/docs", include_in_schema=False)
async def swagger_ui() -> HTMLResponse:
    """The interactive documentation, wearing the shop's own mark."""
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        title=f"{settings.PROJECT_NAME} — API",
        swagger_favicon_url=MARK,
    )


@app.get(f"{settings.API_V1_PREFIX}/redoc", include_in_schema=False)
async def redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        title=f"{settings.PROJECT_NAME} — API",
        redoc_favicon_url=MARK,
    )


def branded_openapi() -> dict[str, Any]:
    """The schema, with the mark ReDoc renders in its sidebar.

    Cached on the app the way FastAPI caches its own, so the schema is built
    once rather than on every request for the docs page.
    """
    if app.openapi_schema is None:
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=TAGS,
        )
        schema["info"]["x-logo"] = {"url": LOGO, "altText": settings.PROJECT_NAME}
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = branded_openapi  # type: ignore[method-assign]

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["system"])
app.include_router(
    auth_router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["authentication"]
)
app.include_router(
    support_router, prefix=f"{settings.API_V1_PREFIX}/support", tags=["support"]
)
app.include_router(
    cases_router, prefix=f"{settings.API_V1_PREFIX}/support/cases", tags=["staff"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
