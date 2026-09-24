"""Vector storage and retrieval backed by PostgreSQL and pgvector.

Search is an exact nearest-neighbour scan. pgvector's approximate indexes (HNSW and IVFFlat)
accept at most 2,000 dimensions and this corpus is embedded at 2,048, but at a few hundred
chunks an exact scan is both faster and perfectly accurate. See the README for the route to
an approximate index if the corpus ever outgrows that.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Engine, Integer, MetaData, Table, Text, func, select, text
from sqlalchemy.dialects.postgresql import insert

from app.corpus.models import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class ChunkStore:
    def __init__(self, engine: Engine, dimensions: int, table_name: str = "chunks") -> None:
        self._engine = engine
        self._metadata = MetaData()
        self._table = Table(
            table_name,
            self._metadata,
            Column("chunk_id", Text, primary_key=True),
            Column("source", Text, nullable=False),
            Column("chapter", Text),
            Column("section", Text),
            Column("article", Text),
            Column("article_title", Text),
            Column("paragraph", Text),
            Column("text", Text, nullable=False),
            Column("body", Text, nullable=False),
            Column("char_count", Integer, nullable=False),
            Column("embedding", Vector(dimensions), nullable=False),
        )

    def create_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self._metadata.create_all(conn)

    def drop_schema(self) -> None:
        with self._engine.begin() as conn:
            self._metadata.drop_all(conn)

    def count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(self._table)).scalar_one()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert or replace chunks, so re-ingesting a source is idempotent."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "article": chunk.article,
                "article_title": chunk.article_title,
                "paragraph": chunk.paragraph,
                "text": chunk.text,
                "body": chunk.body,
                "char_count": chunk.char_count,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        statement = insert(self._table)
        statement = statement.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={c: statement.excluded[c] for c in rows[0] if c != "chunk_id"},
        )

        with self._engine.begin() as conn:
            conn.execute(statement, rows)
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        source: str | None = None,
    ) -> list[Retrieved]:
        distance = self._table.c.embedding.cosine_distance(query_vector)
        statement = select(self._table, distance.label("distance")).order_by(distance).limit(limit)
        if source is not None:
            statement = statement.where(self._table.c.source == source)

        with self._engine.connect() as conn:
            rows = conn.execute(statement).mappings().all()

        return [
            Retrieved(
                chunk=Chunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    body=row["body"],
                    source=row["source"],
                    chapter=row["chapter"],
                    section=row["section"],
                    article=row["article"],
                    article_title=row["article_title"],
                    paragraph=row["paragraph"],
                ),
                score=1.0 - float(row["distance"]),
            )
            for row in rows
        ]
