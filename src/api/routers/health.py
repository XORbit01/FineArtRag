"""
Health and runtime status routes.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_rag_service
from src.api.schemas import HealthResponse
from src.api.service import RAGApiService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: RAGApiService = Depends(get_rag_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        initialized=service.is_initialized(),
    )

