"""
Pydantic schemas for HTTP/WebSocket API payloads.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    source_file: str = "unknown"
    url: Optional[str] = None
    preview: str = ""


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    return_sources: bool = True
    use_memory: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    initialized: bool


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    created_at: float
    updated_at: float
    turn_count: int
    summary_present: bool


class ChatDetail(BaseModel):
    chat_id: str
    title: str
    created_at: float
    updated_at: float
    summary: str = ""
    turns: List[dict] = Field(default_factory=list)
