# ─────────────────────────────────────────────────────────────────────────────
# app/exemplars.py  —  Episodic Reflection Memory & Adaptive Few-Shot Store
#
# Part of the Self-Learning Loop (Step 3):
#   1. Episodic Exemplar Memory: Stores corrective feedback, ground-truth
#      demonstrations, and resolved failures with vector embeddings.
#   2. Dynamic In-Context Adaptation: Semantically matches incoming user queries
#      against learned feedback exemplars to inject runtime prompt guidance.
#   3. Failure Topic Clustering: Clusters historical failures via semantic
#      embeddings to detect blind spots in retrieval and knowledge bases.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EXEMPLARS_PATH,
    FAILED_QUERIES,
    EXEMPLAR_SIMILARITY_THRESHOLD,
    MAX_EXEMPLARS_IN_PROMPT,
)
from app.ingest import get_model
from app.logger import logger

# In-memory cache of exemplar records and their vectors
_cached_exemplars: Optional[list[dict]] = None
_cached_vectors: Optional[np.ndarray] = None


def _cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a 1D query vector and a 2D matrix of vectors."""
    dot = np.dot(doc_vecs, query_vec)
    norm_q = np.linalg.norm(query_vec)
    norm_docs = np.linalg.norm(doc_vecs, axis=1)
    # Avoid zero division
    denominator = norm_docs * norm_q
    denominator[denominator == 0] = 1e-9
    return dot / denominator


def load_exemplars() -> list[dict]:
    """Load exemplars from SQLite database (with fallback to JSON)."""
    global _cached_exemplars, _cached_vectors
    if _cached_exemplars is None:
        try:
            from app.db import get_all_exemplars_db
            _cached_exemplars = get_all_exemplars_db()
        except Exception as e:
            logger.debug(f"Database exemplar load fallback to JSON: {e}")
            _cached_exemplars = None

        if _cached_exemplars is None:
            if not EXEMPLARS_PATH.exists():
                _cached_exemplars = []
            else:
                try:
                    with open(EXEMPLARS_PATH, "r", encoding="utf-8") as f:
                        _cached_exemplars = json.load(f)
                except Exception:
                    _cached_exemplars = []

        if _cached_exemplars:
            vectors = []
            model = None
            for item in _cached_exemplars:
                emb = item.get("embedding")
                if emb:
                    vectors.append(emb)
                else:
                    if model is None:
                        model = get_model()
                    new_emb = model.encode(item["query"], convert_to_numpy=True).tolist()
                    item["embedding"] = new_emb
                    vectors.append(new_emb)
            _cached_vectors = np.array(vectors, dtype=np.float32)
        else:
            _cached_vectors = np.empty((0, 384), dtype=np.float32)

    return _cached_exemplars


def save_exemplars(exemplars: list[dict]) -> None:
    """Save exemplars to disk JSON and update in-memory cache."""
    global _cached_exemplars, _cached_vectors
    EXEMPLARS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXEMPLARS_PATH, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, indent=2, ensure_ascii=False)

    _cached_exemplars = exemplars
    if exemplars:
        vectors = [item["embedding"] for item in exemplars if "embedding" in item]
        _cached_vectors = np.array(vectors, dtype=np.float32) if vectors else np.empty((0, 384), dtype=np.float32)
    else:
        _cached_vectors = np.empty((0, 384), dtype=np.float32)


def add_exemplar(
    query: str,
    answer: str,
    correction: str,
    failure_reason: Optional[str] = None,
    context_summary: Optional[str] = None,
) -> dict:
    """
    Register a newly learned corrective exemplar into SQLite database and memory.
    """
    model = get_model()
    embedding = model.encode(query, convert_to_numpy=True).tolist()
    rec_id = str(uuid.uuid4())

    # 1. Persist to SQLite
    try:
        from app.db import upsert_exemplar_db
        upsert_exemplar_db(
            record_id=rec_id,
            query=query,
            answer=answer,
            correction=correction,
            failure_reason=failure_reason or "user_correction",
            context_summary=context_summary or "",
            embedding=embedding,
        )
    except Exception as e:
        logger.warning(f"Could not persist exemplar to SQLite: {e}")

    # 2. Update memory and JSON
    exemplars = load_exemplars()
    existing = next((e for e in exemplars if e["query"].strip().lower() == query.strip().lower()), None)
    if existing:
        existing["correction"] = correction
        existing["failure_reason"] = failure_reason or existing.get("failure_reason")
        existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
        existing["embedding"] = embedding
        save_exemplars(exemplars)
        logger.info(f"Updated feedback exemplar in SQLite/memory: '{query[:50]}'")
        return existing

    record = {
        "id": rec_id,
        "query": query,
        "answer": answer,
        "correction": correction,
        "failure_reason": failure_reason or "user_correction",
        "context_summary": context_summary or "",
        "embedding": embedding,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    exemplars.append(record)
    save_exemplars(exemplars)
    logger.info(f"Learned new exemplar in SQLite/memory: '{query[:50]}' [Total: {len(exemplars)}]")
    return record


def retrieve_relevant_exemplars(
    query: str,
    top_k: int = MAX_EXEMPLARS_IN_PROMPT,
    threshold: float = EXEMPLAR_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Retrieve semantically matching feedback exemplars for an incoming query.
    Used for runtime in-context prompt adaptation.
    """
    global _cached_vectors
    exemplars = load_exemplars()
    if not exemplars or _cached_vectors is None or len(_cached_vectors) == 0:
        return []

    model = get_model()
    query_vec = model.encode(query, convert_to_numpy=True)

    sims = _cosine_similarity_matrix(query_vec, _cached_vectors)

    matches = []
    for idx, sim in enumerate(sims):
        if sim >= threshold:
            item = dict(exemplars[idx])
            # Drop heavy vector from returned dict for clean logging
            item.pop("embedding", None)
            item["similarity"] = round(float(sim), 4)
            matches.append(item)

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    selected = matches[:top_k]
    if selected:
        logger.info(f"Found {len(selected)} relevant feedback exemplars for query: '{query[:50]}'")
    return selected


def check_query_failure_history(
    query: str,
    threshold: float = 0.60,
) -> dict:
    """
    Determine if the current query is semantically similar to any query that has
    historically received negative feedback in SQLite / failed_queries.
    Used by the Adaptive Retrieval Policy.
    """
    failed_list = []
    try:
        from app.db import get_all_failed_queries_db
        failed_list = get_all_failed_queries_db()
    except Exception:
        pass

    if not failed_list and FAILED_QUERIES.exists():
        try:
            with open(FAILED_QUERIES, "r", encoding="utf-8") as f:
                failed_list = json.load(f)
        except Exception:
            failed_list = []

    if not failed_list:
        return {"has_failure_history": False, "failure_count": 0}

    # Encode query and failed queries
    model = get_model()
    query_vec = model.encode(query, convert_to_numpy=True)
    failed_texts = [f["query"] for f in failed_list]
    failed_vecs = model.encode(failed_texts, convert_to_numpy=True)

    sims = _cosine_similarity_matrix(query_vec, failed_vecs)
    max_idx = int(np.argmax(sims))
    max_sim = float(sims[max_idx])

    if max_sim >= threshold:
        matched_failure = failed_list[max_idx]
        return {
            "has_failure_history": True,
            "failure_count": matched_failure.get("failure_count", 1),
            "matched_query": matched_failure["query"],
            "similarity": round(max_sim, 4),
        }

    return {"has_failure_history": False, "failure_count": 0}


def cluster_failures(distance_threshold: float = 0.35) -> list[dict]:
    """
    Cluster all logged failed queries by semantic similarity (using cosine distance).
    Groups queries with cosine similarity >= (1 - distance_threshold).

    Returns:
        List of failure clusters with theme, member queries, total failure count,
        and recommended remediation.
    """
    failed_list = []
    try:
        from app.db import get_all_failed_queries_db
        failed_list = get_all_failed_queries_db()
    except Exception:
        pass

    if not failed_list and FAILED_QUERIES.exists():
        try:
            with open(FAILED_QUERIES, "r", encoding="utf-8") as f:
                failed_list = json.load(f)
        except Exception:
            failed_list = []

    if not failed_list:
        return []

    if len(failed_list) == 1:
        f = failed_list[0]
        return [{
            "cluster_id": 1,
            "theme": f["query"],
            "total_failures": f.get("failure_count", 1),
            "queries": [f["query"]],
            "recommended_action": "Inspect domain coverage in knowledge base for this topic.",
        }]

    model = get_model()
    texts = [f["query"] for f in failed_list]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Leader-follower clustering using pairwise cosine distance
    clusters: list[list[int]] = []
    for i, emb in enumerate(embeddings):
        assigned = False
        for cluster in clusters:
            centroid_emb = embeddings[cluster[0]]
            # Cosine similarity
            dot = np.dot(emb, centroid_emb)
            norm = np.linalg.norm(emb) * np.linalg.norm(centroid_emb)
            sim = float(dot / norm) if norm > 0 else 0.0
            if (1.0 - sim) <= distance_threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    # Format structured cluster reports
    cluster_reports = []
    for c_id, member_indices in enumerate(clusters, start=1):
        member_queries = [failed_list[idx]["query"] for idx in member_indices]
        total_failures = sum(failed_list[idx].get("failure_count", 1) for idx in member_indices)
        theme = failed_list[member_indices[0]]["query"]

        if total_failures >= 4:
            action = f"Critical Knowledge Gap: Add specialized documents regarding '{theme}'."
        elif total_failures >= 2:
            action = f"Moderate Issue: Add few-shot exemplar corrections or re-phrase query prompt for '{theme}'."
        else:
            action = "Monitor: Single failure detected, observing for recurring patterns."

        cluster_reports.append({
            "cluster_id": c_id,
            "theme": theme,
            "total_failures": total_failures,
            "queries": member_queries,
            "recommended_action": action,
        })

    cluster_reports.sort(key=lambda x: x["total_failures"], reverse=True)
    return cluster_reports
