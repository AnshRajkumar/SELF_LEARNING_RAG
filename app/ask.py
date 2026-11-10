# ─────────────────────────────────────────────────────────────────────────────
# app/ask.py  —  Full RAG pipeline (retrieve → generate → return)
#
# This is the main endpoint users will call.
# It chains together:
#   1. Retrieve: find top-k relevant documents from FAISS
#   2. Generate: produce a natural language answer using those docs
#   3. Auto-evaluate: score the answer quality automatically
#   4. Return: structured response with answer + sources + query ID
#
# Endpoint: POST /ask
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import sys
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_TOP_K
from app.retrieve import search
from app.generate import generate_answer
from app.evaluate import evaluate_response
from app.logger import logger

router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Your question")
    top_k: int = Field(
        DEFAULT_TOP_K, ge=1, le=20,
        description="Number of documents to retrieve for context"
    )


class AskResponse(BaseModel):
    query_id:    str             # Unique ID for this query (use in /feedback)
    query:       str
    answer:      str
    sources:     list[dict]      # Retrieved docs used as context
    auto_eval:   dict            # Automatic quality evaluation
    timestamp:   str
    pipeline:    dict            # Metadata about the pipeline run


# ── Full RAG pipeline ──────────────────────────────────────────────────────

def run_rag_pipeline(query: str, top_k: int) -> AskResponse:
    """
    Execute the full RAG pipeline for a given query.

    Steps:
      1. RETRIEVE  — semantic search via FAISS
      2. GENERATE  — LLM answer from retrieved context
      3. EVALUATE  — auto-score the answer quality
      4. PACKAGE   — wrap everything into a structured response

    Args:
        query: The user's question.
        top_k: Number of documents to retrieve.

    Returns:
        AskResponse with answer, sources, evaluation, and metadata.
    """
    query_id  = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    logger.info(f"[{query_id}] RAG pipeline started | query='{query}' | top_k={top_k}")

    # ── Step 1: Retrieve ──────────────────────────────────────────────
    t0 = datetime.utcnow()
    context_docs = search(query=query, top_k=top_k)
    retrieve_ms  = (datetime.utcnow() - t0).total_seconds() * 1000

    if not context_docs:
        logger.warning(f"[{query_id}] No documents retrieved for query: '{query}'")

    logger.info(f"[{query_id}] Retrieved {len(context_docs)} docs in {retrieve_ms:.1f}ms")

    # ── Step 2: Generate ──────────────────────────────────────────────
    t1 = datetime.utcnow()
    answer      = generate_answer(query=query, context_docs=context_docs)
    generate_ms = (datetime.utcnow() - t1).total_seconds() * 1000

    logger.info(f"[{query_id}] Answer generated in {generate_ms:.1f}ms")

    # ── Step 3: Auto-evaluate (RAG Triad: Faithfulness, Relevancy, Precision) ──
    auto_eval = evaluate_response(
        query=query,
        answer=answer,
        context=context_docs,
    )

    logger.info(
        f"[{query_id}] Auto-eval: score={auto_eval['score']:.3f}, "
        f"label={auto_eval['label']}"
    )

    # ── Step 4: Package response ──────────────────────────────────────
    pipeline_meta = {
        "retrieve_ms":    round(retrieve_ms, 1),
        "generate_ms":    round(generate_ms, 1),
        "total_ms":       round(retrieve_ms + generate_ms, 1),
        "docs_retrieved": len(context_docs),
    }

    logger.info(
        f"[{query_id}] Pipeline complete in {pipeline_meta['total_ms']}ms | "
        f"label={auto_eval['label']}"
    )

    # Persist structured query run trace & auto-eval to SQLite
    try:
        from app.db import log_pipeline_run
        log_pipeline_run(
            query_id=query_id,
            query=query,
            pipeline_meta=pipeline_meta,
            auto_eval=auto_eval,
        )
    except Exception as e:
        logger.warning(f"Failed to log query trace to database: {e}")

    return AskResponse(
        query_id=query_id,
        query=query,
        answer=answer,
        sources=context_docs,
        auto_eval=auto_eval,
        timestamp=timestamp,
        pipeline=pipeline_meta,
    )


# ── API Endpoint ───────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, summary="Full RAG pipeline: query → retrieve → generate → answer")
def ask(request: AskRequest):
    """
    **The main endpoint** — runs the complete Self-Learning RAG pipeline.

    1. 🔍 **Retrieve** relevant documents from the knowledge base
    2. 🤖 **Generate** a natural language answer using those documents
    3. 📊 **Evaluate** the answer quality automatically
    4. 📋 **Return** the answer with sources and metadata

    After receiving the answer, use the `query_id` to submit feedback via:
    ```
    POST /feedback  { "query_id": "...", "feedback": "good" | "bad" }
    ```

    **Example request:**
    ```json
    { "query": "What is retrieval-augmented generation?", "top_k": 3 }
    ```
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return run_rag_pipeline(query=request.query, top_k=request.top_k)
