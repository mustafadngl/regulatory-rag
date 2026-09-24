from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_rag_service
from app.rag import RagService
from app.upstream import UpstreamError

router = APIRouter(tags=["question answering"])


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=1000, examples=["Which AI practices are prohibited?"]
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    citation: str
    chunk_id: str
    article: str | None
    article_title: str | None
    chapter: str | None
    score: float


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(description="False when the extracts did not support an answer.")
    citations: list[CitationResponse]
    model: str
    usage: UsageResponse


@router.post("/ask", response_model=AskResponse, summary="Answer a question with citations")
def ask(
    request: AskRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> AskResponse:
    try:
        result = service.answer(request.question, top_k=request.top_k)
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=f"Model provider unavailable: {exc}") from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        citations=[CitationResponse(**vars(c)) for c in result.citations],
        model=result.model,
        usage=UsageResponse(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieval_ms=round(result.retrieval_ms, 1),
            generation_ms=round(result.generation_ms, 1),
        ),
    )
