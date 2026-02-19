"""
Chat history management routes.
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_rag_service
from src.api.schemas import ChatCreateRequest, ChatDetail, ChatSummary, QueryRequest, QueryResponse
from src.api.service import RAGApiService

router = APIRouter(prefix="/v1/chats", tags=["chats"])


@router.get("", response_model=list[ChatSummary])
def list_chats(service: RAGApiService = Depends(get_rag_service)):
    return service.list_chats()


@router.post("", response_model=ChatSummary)
def create_chat(payload: ChatCreateRequest, service: RAGApiService = Depends(get_rag_service)):
    return service.create_chat(title=payload.title)


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: str, service: RAGApiService = Depends(get_rag_service)):
    return service.get_chat(chat_id)


@router.post("/{chat_id}/reset")
def reset_chat(chat_id: str, service: RAGApiService = Depends(get_rag_service)):
    service.reset_session(chat_id)
    return {"status": "ok"}


@router.post("/{chat_id}/query", response_model=QueryResponse)
def chat_query(chat_id: str, payload: QueryRequest, service: RAGApiService = Depends(get_rag_service)):
    result = service.chat(chat_id, payload.question, return_sources=payload.return_sources)
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@router.delete("/{chat_id}")
def delete_chat(chat_id: str, service: RAGApiService = Depends(get_rag_service)):
    deleted = service.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "ok"}
