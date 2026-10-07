"""
backend/api/rag.py
==================
FastAPI router for the maintenance knowledge base RAG endpoints.

Endpoints
---------
POST /api/v1/rag/query   — retrieve relevant maintenance guidance for a question.
POST /api/v1/rag/ingest  — (re-)build the vector store (protected by API key).
GET  /api/v1/rag/status  — report whether the knowledge base is ready.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from backend.core.config import Settings, get_settings
from rag.retriever import RetrievalResult
from rag.service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _require_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Dependency that rejects requests without a valid API key."""
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Maintenance or troubleshooting question.",
        examples=["What are the signs of bearing spalling?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of knowledge chunks to return.",
    )


class RetrievedChunk(BaseModel):
    content: str
    source: str
    chunk_index: int
    distance: float = Field(description="Cosine distance — lower means more relevant.")


class QueryResponse(BaseModel):
    question: str
    results: list[RetrievedChunk]
    total_results: int


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


class StatusResponse(BaseModel):
    ready: bool
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/rag/query",
    response_model=QueryResponse,
    summary="Query the maintenance knowledge base",
    description=(
        "Retrieve the most relevant maintenance guidance chunks for a given question "
        "using semantic similarity search over the embedded knowledge base."
    ),
)
def query_knowledge_base(
    body: QueryRequest,
    svc: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    results: list[RetrievalResult] = svc.query(question=body.question, top_k=body.top_k)
    chunks = [
        RetrievedChunk(
            content=r.content,
            source=r.source,
            chunk_index=r.chunk_index,
            distance=r.distance,
        )
        for r in results
    ]
    return QueryResponse(
        question=body.question,
        results=chunks,
        total_results=len(chunks),
    )


@router.post(
    "/rag/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="(Re-)ingest the knowledge base",
    description=(
        "Load all markdown documents from the knowledge base, embed them, "
        "and persist to the vector store.  Requires a valid X-API-Key header."
    ),
    dependencies=[Depends(_require_api_key)],
)
def ingest_knowledge_base(
    svc: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    count = svc.ingest()
    return IngestResponse(
        chunks_indexed=count,
        message=f"Knowledge base ingested successfully — {count} chunk(s) indexed.",
    )


@router.get(
    "/rag/status",
    response_model=StatusResponse,
    summary="Knowledge base readiness",
    description="Returns whether the knowledge base has been ingested and is ready to query.",
)
def rag_status(
    svc: RAGService = Depends(get_rag_service),
) -> StatusResponse:
    ready = svc.is_ready
    return StatusResponse(
        ready=ready,
        message="Knowledge base is ready." if ready else "Knowledge base not yet ingested.",
    )
