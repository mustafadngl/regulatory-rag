"""Retrieval-augmented answering.

The prompt is deliberately strict. A model that answers plausibly from its own training data
instead of the retrieved extracts produces exactly the failure this project exists to measure,
so refusing is treated as a correct outcome rather than a shortcoming.
"""

import time
from dataclasses import dataclass

from app.corpus.embeddings import EmbeddingClient
from app.corpus.store import ChunkStore, Retrieved
from app.generation import ChatClient
from app.text import normalise_typography

UNANSWERABLE = "The provided extracts do not answer this question."

SYSTEM_PROMPT = f"""You answer questions about European Union regulatory texts.

Rules:
- Use only the numbered extracts provided. Never rely on prior knowledge of the law.
- Cite the article supporting each claim, written as (Article 6(2)).
- If the extracts do not contain the answer, reply with exactly: {UNANSWERABLE}
- Answer in no more than 60 words. Do not speculate and do not give legal advice.
- Write plain ASCII punctuation. Use ordinary spaces and hyphens."""


@dataclass
class Citation:
    citation: str
    chunk_id: str
    article: str | None
    article_title: str | None
    chapter: str | None
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    model: str
    grounded: bool
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float


def build_context(hits: list[Retrieved]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        heading = hit.chunk.citation
        if hit.chunk.article_title:
            heading = f"{heading} — {hit.chunk.article_title}"
        blocks.append(f"[{index}] {heading}\n{hit.chunk.body}")
    return "\n\n".join(blocks)


class RagService:
    def __init__(
        self,
        store: ChunkStore,
        embedder: EmbeddingClient,
        chat: ChatClient,
        *,
        top_k: int = 5,
        source: str | None = None,
        system_prefix: str = "",
        max_answer_tokens: int = 400,
    ) -> None:
        self._max_answer_tokens = max_answer_tokens
        self._store = store
        self._embedder = embedder
        self._chat = chat
        self._top_k = top_k
        self._source = source
        self._system_prompt = (
            f"{system_prefix}\n\n{SYSTEM_PROMPT}" if system_prefix else SYSTEM_PROMPT
        )

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        query_vector = self._embedder.embed_query(question)
        hits = self._store.search(
            query_vector,
            limit=top_k or self._top_k,
            source=self._source,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000

        citations = [
            Citation(
                citation=hit.chunk.citation,
                chunk_id=hit.chunk.chunk_id,
                article=hit.chunk.article,
                article_title=hit.chunk.article_title,
                chapter=hit.chunk.chapter,
                score=round(hit.score, 4),
            )
            for hit in hits
        ]

        if not hits:
            return Answer(
                question=question,
                answer=UNANSWERABLE,
                citations=[],
                model=self._chat.model,
                grounded=False,
                prompt_tokens=0,
                completion_tokens=0,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )

        prompt = f"Extracts:\n\n{build_context(hits)}\n\nQuestion: {question}"
        generation_started = time.perf_counter()
        completion = self._chat.complete(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._max_answer_tokens,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000

        answer = normalise_typography(completion.content)
        grounded = UNANSWERABLE.lower() not in answer.lower()
        return Answer(
            question=question,
            answer=answer,
            citations=citations if grounded else [],
            model=self._chat.model,
            grounded=grounded,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
