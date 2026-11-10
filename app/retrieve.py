# ─────────────────────────────────────────────────────────────────────────────
# app/retrieve.py  —  Hybrid Retrieval (BM25 + FAISS) with RRF & Re-ranking
#
# Implements:
#   1. Dense Vector Search: FAISS index with sentence-transformers embeddings
#   2. Sparse Lexical Search: Okapi BM25 for precise keyword matching
#   3. Reciprocal Rank Fusion (RRF): Combines dense + sparse rankings:
#        RRF(d) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
#   4. Re-ranking: Cross-Encoder / LLM judge to re-score candidate passages
#
# Endpoint: POST /retrieve
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import sys
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    DEFAULT_TOP_K,
    RETRIEVAL_MODE,
    RRF_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    CANDIDATE_FACTOR,
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from app.ingest import get_model
from app.bm25 import BM25Okapi
from app.logger import logger

router = APIRouter()

# ── Global Cached States ───────────────────────────────────────────────────
_index: Optional[faiss.Index] = None
_metadata: Optional[list[dict]] = None
_bm25: Optional[BM25Okapi] = None
_cross_encoder = None
_cross_encoder_failed = False


def load_index() -> faiss.Index:
    """Load FAISS index from disk. Cached in memory after first load."""
    global _index
    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not found. Call POST /ingest first."
            )
        logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index loaded. Vectors: {_index.ntotal}")
    return _index


def load_metadata() -> list[dict]:
    """Load document metadata (text + source). Cached after first load."""
    global _metadata
    if _metadata is None:
        if not METADATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Metadata not found. Call POST /ingest first."
            )
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        logger.info(f"Metadata loaded: {len(_metadata)} chunks")
    return _metadata


def load_bm25() -> BM25Okapi:
    """Initialize or retrieve in-memory BM25 index constructed from metadata."""
    global _bm25
    if _bm25 is None:
        metadata = load_metadata()
        corpus = [doc.get("text", "") for doc in metadata]
        logger.info(f"Building Okapi BM25 index across {len(corpus)} documents...")
        _bm25 = BM25Okapi(corpus=corpus)
        logger.info("BM25 index built successfully.")
    return _bm25


def reload_index() -> None:
    """Force reload of FAISS index, metadata, and BM25 index (call after re-ingest)."""
    global _index, _metadata, _bm25
    _index = None
    _metadata = None
    _bm25 = None
    logger.info("FAISS index, metadata, and BM25 caches cleared.")


# ── Dense Vector Retrieval ─────────────────────────────────────────────────

def dense_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Perform dense semantic vector retrieval via FAISS."""
    model = get_model()
    index = load_index()
    metadata = load_metadata()

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1 or idx >= len(metadata):
            continue
        doc = metadata[idx]
        results.append({
            "idx": idx,
            "text": doc["text"],
            "source": doc["source"],
            "score": float(round(score, 4)),
            "dense_score": float(round(score, 4)),
            "rank": rank,
            "channel": "dense",
        })
    return results


# ── Sparse Lexical Retrieval ───────────────────────────────────────────────

def sparse_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Perform sparse lexical keyword retrieval via Okapi BM25."""
    bm25 = load_bm25()
    metadata = load_metadata()

    bm25_matches = bm25.search(query, top_k=top_k)

    results = []
    for rank, (idx, score) in enumerate(bm25_matches, start=1):
        if idx >= len(metadata):
            continue
        doc = metadata[idx]
        results.append({
            "idx": idx,
            "text": doc["text"],
            "source": doc["source"],
            "score": float(round(score, 4)),
            "sparse_score": float(round(score, 4)),
            "rank": rank,
            "channel": "sparse",
        })
    return results


# ── Reciprocal Rank Fusion (RRF) ───────────────────────────────────────────

def fuse_results_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    k_rrf: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[dict]:
    """
    Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula for document d:
      RRF_Score(d) = dense_weight / (k_rrf + rank_dense) + sparse_weight / (k_rrf + rank_sparse)
    """
    fused_scores: dict[int, float] = {}
    doc_map: dict[int, dict] = {}

    # Accumulate dense contributions
    for item in dense_results:
        idx = item["idx"]
        doc_map[idx] = item
        rank = item["rank"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (dense_weight / (k_rrf + rank))

    # Accumulate sparse contributions
    for item in sparse_results:
        idx = item["idx"]
        if idx not in doc_map:
            doc_map[idx] = item
        else:
            # Preserve sparse score details in existing item
            doc_map[idx]["sparse_score"] = item.get("sparse_score")
        rank = item["rank"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (sparse_weight / (k_rrf + rank))

    # Sort descending by fused RRF score
    sorted_indices = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)

    fused_results = []
    for rank, idx in enumerate(sorted_indices, start=1):
        item = dict(doc_map[idx])
        item["rrf_score"] = float(round(fused_scores[idx], 5))
        item["score"] = item["rrf_score"]
        item["rank"] = rank
        item["channel"] = "hybrid_rrf"
        fused_results.append(item)

    return fused_results


# ── Cross-Encoder & LLM Re-ranking ─────────────────────────────────────────

def get_cross_encoder():
    """Lazily load Cross-Encoder model with error recovery."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is None and not _cross_encoder_failed:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {RERANKER_MODEL}")
            _cross_encoder = CrossEncoder(RERANKER_MODEL)
            logger.info("CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder ({e}). Will use LLM / RRF fallback.")
            _cross_encoder_failed = True
    return _cross_encoder


def _rerank_with_cross_encoder(query: str, candidates: list[dict]) -> Optional[list[dict]]:
    """Score (query, text) pairs using CrossEncoder."""
    encoder = get_cross_encoder()
    if encoder is None:
        return None

    try:
        pairs = [[query, doc["text"]] for doc in candidates]
        raw_scores = encoder.predict(pairs)

        for i, s in enumerate(raw_scores):
            # Sigmoid normalization for raw logits
            prob = float(1.0 / (1.0 + np.exp(-float(s))))
            candidates[i]["rerank_score"] = round(prob, 4)
            candidates[i]["score"] = candidates[i]["rerank_score"]

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
    except Exception as e:
        logger.warning(f"CrossEncoder prediction failed: {e}")
        return None


def _rerank_with_llm(query: str, candidates: list[dict]) -> Optional[list[dict]]:
    """Pointwise/Listwise relevance scoring using Groq LLM."""
    if LLM_PROVIDER != "groq" or not GROQ_API_KEY:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        docs_text = "\n\n".join(
            f"[ID {i}]: {doc['text'][:250]}" for i, doc in enumerate(candidates)
        )

        prompt = (
            "You are a search relevance re-ranker. Given the query and documents below, "
            "assign a relevance score between 0.0 (irrelevant) and 1.0 (highly relevant) to each document.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{docs_text}\n\n"
            'Return ONLY a valid JSON object in this format:\n{"scores": [score0, score1, ...]}'
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=250,
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        scores = data.get("scores", [])

        if len(scores) == len(candidates):
            for i, s in enumerate(scores):
                s_val = round(max(0.0, min(1.0, float(s))), 4)
                candidates[i]["rerank_score"] = s_val
                candidates[i]["score"] = s_val
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates
    except Exception as e:
        logger.warning(f"LLM reranker failed ({e}). Retaining prior order.")
    return None


def re_rank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-rank candidate documents using Cross-Encoder or fast LLM judge.
    Falls back gracefully to RRF order if reranker is unavailable.
    """
    if not results:
        return results

    logger.info(f"Re-ranking {len(results)} candidate documents for: '{query[:50]}'...")

    # Strategy 1: Cross-Encoder model
    reranked = _rerank_with_cross_encoder(query, results)
    rerank_engine = "cross_encoder"

    # Strategy 2: LLM judge reranker
    if reranked is None:
        reranked = _rerank_with_llm(query, results)
        rerank_engine = "llm_judge" if reranked is not None else "rrf_fallback"

    # Strategy 3: Retain RRF rank
    if reranked is None:
        reranked = results

    # Re-index ranks
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
        item["rerank_engine"] = rerank_engine

    logger.info(f"Re-ranking complete via {rerank_engine}.")
    return reranked


# ── Unified Search Pipeline ────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    mode: str = RETRIEVAL_MODE,
    use_reranker: bool = RERANKER_ENABLED,
) -> list[dict]:
    """
    Execute the unified retrieval pipeline:
      1. Dense, Sparse, or Hybrid (RRF) candidate retrieval
      2. Re-ranking via Cross-Encoder / LLM judge
      3. Return top-k relevant documents with normalized scores

    Args:
        query: User search question.
        top_k: Number of final results to return.
        mode: "hybrid" (default), "dense", or "sparse".
        use_reranker: Whether to apply the re-ranking stage.

    Returns:
        List of dicts: [{"text": ..., "source": ..., "score": ..., "rank": ...}]
    """
    logger.info(f"Retrieving for '{query}' | mode='{mode}' | top_k={top_k} | rerank={use_reranker}")

    # Step 1: Adaptive Retrieval Policy (Step 3 Self-Learning)
    factor = CANDIDATE_FACTOR
    adaptive_triggered = False
    try:
        from config import ADAPTIVE_RETRIEVAL_ENABLED
        from app.exemplars import check_query_failure_history
        if ADAPTIVE_RETRIEVAL_ENABLED:
            fail_info = check_query_failure_history(query)
            if fail_info.get("has_failure_history"):
                adaptive_triggered = True
                factor = int(CANDIDATE_FACTOR * 1.5) + 1
                logger.info(
                    f"Adaptive Retrieval Policy triggered: topic has {fail_info.get('failure_count')} "
                    f"prior failures ('{fail_info.get('matched_query')}'). Widened candidate pool to factor {factor}."
                )
    except Exception as e:
        logger.debug(f"Adaptive policy check skipped: {e}")

    # Step 2: Candidate Retrieval
    candidates_count = max(top_k * factor, top_k) if use_reranker else top_k

    if mode == "dense":
        candidates = dense_search(query=query, top_k=candidates_count)
    elif mode == "sparse":
        candidates = sparse_search(query=query, top_k=candidates_count)
    else:  # hybrid
        dense_candidates = dense_search(query=query, top_k=candidates_count)
        sparse_candidates = sparse_search(query=query, top_k=candidates_count)
        candidates = fuse_results_rrf(dense_candidates, sparse_candidates)

    if not candidates:
        logger.warning(f"No documents found for query: '{query}'")
        return []

    # Step 2: Re-ranking
    if use_reranker:
        candidates = re_rank(query=query, results=candidates)

    final_results = candidates[:top_k]

    # Re-normalize final ranks and ensure clean structure
    cleaned_results = []
    for rank, doc in enumerate(final_results, start=1):
        score_val = float(doc.get("score", doc.get("rerank_score", doc.get("rrf_score", 0.0))))
        cleaned_results.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": round(score_val, 4),
            "rank": rank,
            "channel": doc.get("channel", mode),
        })

    logger.info(f"Retrieved {len(cleaned_results)} final results for query: '{query}'")
    return cleaned_results


# ── Pydantic API Models ────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of results to return")
    mode: Optional[Literal["hybrid", "dense", "sparse"]] = Field(
        default=None,
        description="Retrieval mode: 'hybrid' (dense+sparse RRF), 'dense' (FAISS), or 'sparse' (BM25)"
    )
    rerank: Optional[bool] = Field(
        default=None,
        description="Whether to apply cross-encoder / LLM re-ranking (defaults to config setting)"
    )


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    mode: str
    reranked: bool
    results: list[dict]
    total_found: int


# ── API Endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve top-k relevant documents using Hybrid Search (BM25 + FAISS) & Re-ranking"
)
def retrieve_documents(request: RetrieveRequest):
    """
    Hybrid semantic document retrieval:
    - **Dense Semantic Search**: FAISS index with `all-MiniLM-L6-v2` embeddings
    - **Sparse Lexical Search**: Okapi BM25 keyword matching
    - **Reciprocal Rank Fusion (RRF)**: Combines dense + sparse rankings
    - **Re-ranking**: Cross-Encoder / LLM judge to re-score candidate passages
    """
    mode = request.mode or RETRIEVAL_MODE
    rerank = request.rerank if request.rerank is not None else RERANKER_ENABLED

    results = search(
        query=request.query,
        top_k=request.top_k,
        mode=mode,
        use_reranker=rerank,
    )

    return RetrieveResponse(
        query=request.query,
        top_k=request.top_k,
        mode=mode,
        reranked=rerank,
        results=results,
        total_found=len(results),
    )
