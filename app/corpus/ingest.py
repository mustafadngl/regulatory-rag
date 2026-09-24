"""Ingestion pipeline: fetch a regulation, chunk it, embed it and store it.

Run with:  python -m app.corpus.ingest --celex 32024R1689
"""

import argparse
import logging
import time
from pathlib import Path

from app.config import get_settings
from app.corpus.chunking import chunk_articles
from app.corpus.embeddings import EmbeddingClient
from app.corpus.eurlex import fetch_regulation, parse_xhtml
from app.corpus.store import ChunkStore
from app.db import make_engine

logger = logging.getLogger(__name__)

DEFAULT_CELEX = "32024R1689"


def ingest(celex: str, cache_dir: Path, *, recreate: bool = False) -> int:
    settings = get_settings()

    logger.info("Fetching %s", celex)
    html = fetch_regulation(celex, cache_dir, ca_bundle=settings.ca_bundle)

    articles = parse_xhtml(html)
    chunks = chunk_articles(articles, source=celex)
    logger.info("Parsed %d articles into %d chunks", len(articles), len(chunks))

    store = ChunkStore(
        make_engine(settings.database_url),
        dimensions=settings.embedding_dimensions,
    )
    if recreate:
        store.drop_schema()
    store.create_schema()

    started = time.perf_counter()
    with EmbeddingClient(
        settings.llm_api_base,
        settings.llm_api_key,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.request_timeout,
        ca_bundle=settings.ca_bundle,
    ) as client:
        vectors = client.embed_passages([chunk.text for chunk in chunks])
    logger.info("Embedded %d chunks in %.1fs", len(vectors), time.perf_counter() - started)

    stored = store.upsert(chunks, vectors)
    logger.info("Stored %d chunks; table now holds %d", stored, store.count())
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celex", default=DEFAULT_CELEX, help="CELEX identifier to ingest")
    parser.add_argument("--cache-dir", default="data/raw", type=Path)
    parser.add_argument("--recreate", action="store_true", help="drop the table before ingesting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest(args.celex, args.cache_dir, recreate=args.recreate)


if __name__ == "__main__":
    main()
