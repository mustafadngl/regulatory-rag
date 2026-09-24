from fastapi import HTTPException, Request

from app.rag import RagService


def get_rag_service(request: Request) -> RagService:
    service: RagService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Question answering is not configured")
    return service
