"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.embedding import Embedder
from app.agent.inference import HuggingFaceEmbedder
from app.agent.knowledge import load_corpus
from app.agent.retrieval import PolicyIndex
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.core.config import settings
from app.db.session import sessionmanager


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

    yield
    if sessionmanager._engine is not None:
        # Close the DB connection
        await sessionmanager.close()


def _embedder() -> Embedder | None:
    """The hosted embedder when a token is configured, otherwise none."""
    if settings.HUGGINGFACE_API_TOKEN is None:
        return None
    return HuggingFaceEmbedder(settings)


app = FastAPI(
    lifespan=lifespan,
    debug=settings.DEBUG,
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
