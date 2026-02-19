"""
Query routes using RAGSystem through the service layer.
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_rag_service
from src.api.schemas import QueryRequest, QueryResponse
from src.api.service import RAGApiService

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    service: RAGApiService = Depends(get_rag_service),
) -> QueryResponse:
    try:
        result = service.query(
            question=payload.question,
            return_sources=payload.return_sources,
            use_memory=payload.use_memory,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

