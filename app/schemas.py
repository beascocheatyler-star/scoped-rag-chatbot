"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    # Constrained so a bad role is a 422 the client can see, rather than a turn
    # silently dropped from the history.
    role: Literal["user", "assistant"] = Field(
        ..., description="Either 'user' or 'assistant'."
    )
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question.")
    history: List[ChatTurn] = Field(
        default_factory=list, description="Prior turns, oldest first."
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these documents. Omit to search all.",
    )


class CitationModel(BaseModel):
    doc_name: str
    doc_id: str
    chunk_id: str
    position: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool
    decision: str = Field(
        ...,
        description="in_scope | out_of_scope | small_talk | meta | no_corpus",
    )
    citations: List[CitationModel] = Field(default_factory=list)
    retrieval_score: float
    reason: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class DocumentModel(BaseModel):
    doc_id: str
    name: str
    size_bytes: int
    chunk_count: int
    char_count: int
    sha256: str
    uploaded_at: str


class UploadResponse(BaseModel):
    uploaded: List[DocumentModel]
    skipped: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Files that could not be ingested, with the reason why.",
    )
    total_documents: int
    total_chunks: int


class PersonaModel(BaseModel):
    name: str = "Assistant"
    topic: str = "the uploaded documents"
    style: str = "Be clear, concise, and professional."
    greeting: str = ""
    suggested_questions: List[str] = Field(default_factory=list)


class PresetRequest(BaseModel):
    preset: str = Field(..., description="customer_care | course_ta | generic")


class StatusResponse(BaseModel):
    status: str
    llm_configured: bool
    model: str
    documents: int
    chunks: int
    persona: PersonaModel
    greeting: str
    strict_scope: bool
    relevance_floor: float


class ErrorResponse(BaseModel):
    detail: str
