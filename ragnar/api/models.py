from __future__ import annotations

from pydantic import BaseModel


# Request/Response Models

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    strategy: str = 'hierarchical'


class QueryResponse(BaseModel):
    answer: str
    grounding: list[dict]
    grounded_percentage: float
    sources: list[dict]
    session_id: str
    conversation_length: int
    metrics: dict | None = None
    cache_hit: bool = False
    cache_score: float | None = None


class IndexRequest(BaseModel):
    path: str
    strategy: str = 'hierarchical'


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks: int | None = None
    parent_chunks: int | None = None
    child_chunks: int | None = None


class MessageModel(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[MessageModel]
