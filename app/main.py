import logging

from fastapi import FastAPI

from app import __version__
from app.api import health
from app.config import get_settings


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
    )
    app.include_router(health.router)
    return app


app = create_app()
