import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text

from app.corpus.store import ChunkStore
from app.db import make_engine

DEFAULT_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/regrag"
TEST_TABLE = "chunks_integration_test"
DIMENSIONS = 4


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Skip the whole integration suite when no database is reachable."""
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    candidate = make_engine(url)
    try:
        with candidate.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no database at {url}: {type(exc).__name__}")
    return candidate


@pytest.fixture
def store(engine: Engine) -> Iterator[ChunkStore]:
    """A store backed by its own table, so tests never touch ingested data."""
    instance = ChunkStore(engine, dimensions=DIMENSIONS, table_name=TEST_TABLE)
    instance.drop_schema()
    instance.create_schema()
    yield instance
    instance.drop_schema()
