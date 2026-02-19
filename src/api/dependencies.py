"""
Dependency helpers for FastAPI routers.
"""

from fastapi import Request

from src.api.service import RAGApiService


def get_rag_service(request: Request) -> RAGApiService:
    return request.app.state.rag_service

