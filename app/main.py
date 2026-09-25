import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import ask, health
from app.config import Settings, get_settings
from app.corpus.embeddings import EmbeddingClient
from app.corpus.store import ChunkStore
from app.db import make_engine
from app.generation import ChatClient
from app.rag import RagService

logger = logging.getLogger(__name__)


def build_rag_service(settings: Settings) -> RagService:
    store = ChunkStore(
        make_engine(settings.database_url),
        dimensions=settings.embedding_dimensions,
    )
    embedder = EmbeddingClient(
        settings.llm_api_base,
        settings.llm_api_key,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.request_timeout,
        ca_bundle=settings.ca_bundle,
        max_attempts=settings.upstream_max_attempts,
    )
    chat = ChatClient(
        settings.llm_api_base,
        settings.llm_api_key,
        settings.llm_model,
        timeout=settings.request_timeout,
        ca_bundle=settings.ca_bundle,
        max_attempts=settings.upstream_max_attempts,
    )
    return RagService(
        store,
        embedder,
        chat,
        top_k=settings.retrieval_top_k,
        system_prefix=settings.llm_system_prefix,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.llm_api_key:
        app.state.rag_service = build_rag_service(settings)
    else:
        # The service still starts so that health checks and docs work without credentials.
        logger.warning("LLM_API_KEY is not set; /ask will return 503")
        app.state.rag_service = None
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="Regulatory RAG",
        description=(
            "Question answering over EU regulatory texts with cited sources "
            "and automated answer-quality evaluation."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(ask.router)
    return app


app = create_app()
