"""
MediQuery — Pydantic API Models
==================================
Request/response schemas for the clinical RAG decision support API.
"""

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Request Models
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Incoming clinical query from the frontend."""
    question: str
    context: str = ""
    use_retrieval: bool = True
    use_web_search: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Response Models
# ─────────────────────────────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    """Full clinical response with evidence metadata."""
    response: str
    source: str = ""
    model_used: str = ""           # "llama-3.1-8b" or "gemma-4-26b"
    confidence: float = Field(0.0, ge=0.0, le=1.0)  # retrieval confidence
    faithful: bool = True          # NLI entailment check
    cached: bool = False           # was this a cache hit?
    clinical_risk: str = "low"     # "low", "medium", "high"
    cost_usd: float = 0.0         # estimated query cost


class UploadResponse(BaseModel):
    """Response after clinical document upload and indexing."""
    filename: str
    chunks: int
    status: str
    dedup_removed: int = 0         # chunks removed by MinHash dedup


class StatsResponse(BaseModel):
    """Clinical retrieval index statistics."""
    documents: int = 0
    chunks: int = 0
    bm25_terms: int = 0
    cache_entries: int = 0
    total_cost_usd: float = 0.0
    queries_small: int = 0
    queries_big: int = 0


class HealthResponse(BaseModel):
    """System health check."""
    status: str = "ok"
    models_loaded: list[str] = []
    qdrant_connected: bool = False
    redis_connected: bool = False
