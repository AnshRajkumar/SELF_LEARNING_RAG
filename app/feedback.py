# ─────────────────────────────────────────────────────────────────────────────
# app/feedback.py  —  Feedback storage and self-learning loop
#
# This is the heart of the "self-learning" system.
#
# When users submit feedback (good/bad) about a response:
#   - ALL feedback is stored in feedback_log.json for analysis
#   - BAD feedback additionally goes into failed_queries.json
#   - The system surfaces patterns: which queries consistently fail?
#   - Improvement hints are generated to guide future optimizations
#
# Endpoints:
#   POST /feedback              — Submit feedback for a query/answer pair
#   GET  /feedback/stats        — Aggregate statistics (good vs bad counts)
#   GET  /feedback/improvements — Queries that need attention
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEEDBACK_LOG, FAILED_QUERIES, BAD_FEEDBACK_ALERT_THRESHOLD
from app.logger import logger

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────

def _read_json_list(filepath: Path) -> list:
    """Safely read a JSON list from file. Returns empty list if file missing."""
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return []


def _write_json_list(filepath: Path, data: list) -> None:
    """Write a list to a JSON file, creating parent directories as needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Core feedback functions ────────────────────────────────────────────────

def store_feedback(
    query: str,
    context: list[dict],
    answer: str,
    feedback: str,
    auto_score: Optional[float] = None,
    correction: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> dict:
    """
    Store a feedback entry in feedback_log.json.
    If feedback is "bad", also log to failed_queries.json.
    If a correction is provided, saves it as an active reflection exemplar in exemplars.json.

    Args:
        query:          The original user query.
        context:        Retrieved documents that were used as context.
        answer:         The generated answer shown to the user.
        feedback:       "good" or "bad" (user's rating).
        auto_score:     Optional auto-evaluation score from /evaluate.
        correction:     Optional user-provided correction / ground-truth guideline.
        failure_reason: Optional reason category ("hallucination", "incomplete", etc.)

    Returns:
        The saved feedback entry dict (with generated ID + timestamp).
    """
    entry_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat() + "Z"
    sources_list = [doc.get("source", "unknown") for doc in context]
    snippet = context[0]["text"][:200] if context else ""

    # ── SQLite Database Storage (ACID + WAL mode) ──────────────────────
    try:
        from app.db import insert_feedback
        insert_feedback(
            entry_id=entry_id,
            query=query,
            answer=answer,
            rating=feedback,
            auto_score=auto_score,
            correction=correction,
            failure_reason=failure_reason,
            sources=sources_list,
            context_snippet=snippet,
        )
    except Exception as e:
        logger.error(f"Failed to insert feedback into SQLite: {e}")

    entry = {
        "id":             entry_id,
        "timestamp":      ts,
        "query":          query,
        "answer":         answer,
        "feedback":       feedback,
        "auto_score":     auto_score,
        "correction":     correction,
        "failure_reason": failure_reason,
        "sources":        sources_list,
        "context_snippet": snippet,
    }

    # ── Append to feedback log file (retained for file logging) ─────────
    all_feedback = _read_json_list(FEEDBACK_LOG)
    all_feedback.append(entry)
    _write_json_list(FEEDBACK_LOG, all_feedback)
    logger.info(f"Feedback stored [{feedback.upper()}] for query: '{query[:60]}...'")

    # ── If bad, also update failed queries JSON ────────────────────────
    if feedback == "bad":
        _log_failed_query(query, answer, auto_score)

    # ── Active Learning: If correction provided, learn as episodic exemplar ──
    if correction and correction.strip():
        try:
            from app.exemplars import add_exemplar
            add_exemplar(
                query=query,
                answer=answer,
                correction=correction.strip(),
                failure_reason=failure_reason or ("correction" if feedback == "bad" else "positive_demonstration"),
                context_summary=snippet,
            )
            logger.info(f"Learned active correction exemplar for '{query[:50]}'")
        except Exception as e:
            logger.warning(f"Could not record exemplar: {e}")

    return entry


def _log_failed_query(query: str, answer: str, score: Optional[float]) -> None:
    """
    Append a failed query entry to failed_queries.json.
    Tracks how many times each query has failed (for improvement prioritization).
    """
    failed = _read_json_list(FAILED_QUERIES)

    # Check if this exact query was already logged as failed
    existing = next((f for f in failed if f["query"].strip().lower() == query.strip().lower()), None)

    if existing:
        # Increment the failure count for this query
        existing["failure_count"] = existing.get("failure_count", 1) + 1
        existing["last_seen"] = datetime.utcnow().isoformat() + "Z"
        existing["latest_answer"] = answer
    else:
        failed.append({
            "query":         query,
            "failure_count": 1,
            "first_seen":    datetime.utcnow().isoformat() + "Z",
            "last_seen":     datetime.utcnow().isoformat() + "Z",
            "latest_answer": answer,
            "latest_score":  score,
        })

    _write_json_list(FAILED_QUERIES, failed)
    logger.warning(f"Failed query logged: '{query[:60]}'")


def analyze_feedback() -> dict:
    """
    Compute aggregate statistics from all stored feedback.
    Uses SQLite database queries with fallback to flat JSON list.

    Returns:
        Dict with counts, percentages, and per-query breakdowns.
    """
    try:
        from app.db import get_feedback_stats_db
        return get_feedback_stats_db()
    except Exception as e:
        logger.warning(f"Database stats query failed ({e}), falling back to JSON.")

    all_feedback = _read_json_list(FEEDBACK_LOG)
    if not all_feedback:
        return {
            "total":      0,
            "good":       0,
            "bad":        0,
            "good_pct":   0.0,
            "bad_pct":    0.0,
            "avg_score":  None,
            "top_queries": [],
        }

    total = len(all_feedback)
    good  = sum(1 for f in all_feedback if f.get("feedback") == "good")
    bad   = total - good
    scores = [f["auto_score"] for f in all_feedback if f.get("auto_score") is not None]
    avg_score = round(sum(scores) / len(scores), 4) if scores else None
    query_counts = Counter(f["query"] for f in all_feedback)
    top_queries  = [{"query": q, "count": c} for q, c in query_counts.most_common(5)]

    return {
        "total":       total,
        "good":        good,
        "bad":         bad,
        "good_pct":    round(good / total * 100, 1),
        "bad_pct":     round(bad  / total * 100, 1),
        "avg_score":   avg_score,
        "top_queries": top_queries,
    }


def get_improvement_hints() -> list[dict]:
    """
    Identify queries that need the most improvement using indexed SQLite query.

    Returns:
        List of dicts with query, failure_count, and recommended action.
    """
    try:
        from app.db import get_flagged_queries_db
        return get_flagged_queries_db(BAD_FEEDBACK_ALERT_THRESHOLD)
    except Exception as e:
        logger.warning(f"Database flagged query lookup failed ({e}), falling back to JSON.")

    failed = _read_json_list(FAILED_QUERIES)
    urgent = [
        f for f in failed
        if f.get("failure_count", 0) >= BAD_FEEDBACK_ALERT_THRESHOLD
    ]
    urgent.sort(key=lambda x: x.get("failure_count", 0), reverse=True)

    hints = []
    for f in urgent:
        # Suggest what kind of improvement to make
        count = f.get("failure_count", 0)
        if count >= 5:
            action = "High priority: Consider retraining embeddings or adding more documents on this topic."
        elif count >= 3:
            action = "Medium priority: Try prompt engineering or adding targeted documents."
        else:
            action = "Low priority: Monitor for additional failures before taking action."

        hints.append({
            "query":         f["query"],
            "failure_count": count,
            "first_seen":    f.get("first_seen"),
            "last_seen":     f.get("last_seen"),
            "recommended_action": action,
        })

    logger.info(f"Improvement hints generated: {len(hints)} queries flagged.")
    return hints


# ── Stub for future retraining signal ─────────────────────────────────────

def retrain_hint() -> dict:
    """
    [FUTURE HOOK] Signal when the system needs embedding retraining.

    Could be expanded to:
    - Compute query distribution drift
    - Flag when >20% of recent queries have bad feedback
    - Trigger automatic fine-tuning pipelines

    Returns:
        Dict with retraining recommendation and current health metrics.
    """
    stats = analyze_feedback()
    needs_retraining = stats.get("bad_pct", 0) > 20  # More than 20% bad = warn

    return {
        "needs_retraining": needs_retraining,
        "bad_feedback_pct": stats.get("bad_pct", 0),
        "message": (
            "Consider retraining embeddings — high bad feedback rate."
            if needs_retraining
            else "System health looks good. No retraining needed yet."
        ),
    }


# ── Pydantic models ────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The original user query")
    context: list[dict] = Field(
        default=[],
        description="Retrieved context docs (from /retrieve or /ask response)"
    )
    answer: str = Field(..., min_length=1, description="The generated answer")
    feedback: Literal["good", "bad"] = Field(
        ..., description="User's rating: 'good' if answer was helpful, 'bad' if not"
    )
    auto_score: Optional[float] = Field(
        None, description="Optional auto-evaluation score from /evaluate"
    )
    correction: Optional[str] = Field(
        None, description="Optional user correction or ground-truth guidance (saved as an active reflection exemplar)"
    )
    failure_reason: Optional[str] = Field(
        None, description="Optional failure reason: 'hallucination', 'incomplete', 'out_of_domain', 'wrong_retrieval'"
    )


class FeedbackResponse(BaseModel):
    status:             str
    id:                 str
    feedback:           str
    message:            str
    exemplar_recorded:  bool = False


# ── API Endpoints ──────────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse, summary="Submit feedback & active learning corrections")
def submit_feedback(request: FeedbackRequest):
    """
    Record user feedback to power the self-learning loop:
    - **good**: Stored as a positive example for analysis.
    - **bad**: Stored + logged to failed_queries.json for improvement tracking.
    - **correction**: If provided, automatically registered as an active reflection exemplar
      in `exemplars.json`. Future similar queries will retrieve this guidance to adapt the generation prompt!
    """
    entry = store_feedback(
        query=request.query,
        context=request.context,
        answer=request.answer,
        feedback=request.feedback,
        auto_score=request.auto_score,
        correction=request.correction,
        failure_reason=request.failure_reason,
    )

    has_correction = bool(request.correction and request.correction.strip())
    if request.feedback == "good":
        message = "Thank you! Positive feedback recorded. [GOOD]"
    else:
        message = "Feedback recorded. This query has been flagged for improvement. [FLAGGED]"

    if has_correction:
        message += " Learned corrective exemplar registered into episodic memory."

    return FeedbackResponse(
        status="success",
        id=entry["id"],
        feedback=request.feedback,
        message=message,
        exemplar_recorded=has_correction,
    )


@router.get("/feedback/exemplars", summary="List learned feedback reflection exemplars")
def get_exemplars():
    """
    Return all active learned feedback exemplars stored in the reflection memory.
    These are used for runtime prompt adaptation.
    """
    from app.exemplars import load_exemplars
    raw_exemplars = load_exemplars()
    # Strip heavy vector embeddings from response
    cleaned = []
    for ex in raw_exemplars:
        item = dict(ex)
        item.pop("embedding", None)
        cleaned.append(item)
    return {
        "total_exemplars": len(cleaned),
        "exemplars": cleaned,
    }


@router.post("/feedback/optimize", summary="Cluster failed queries & produce self-learning optimizations")
def optimize_from_feedback():
    """
    Cluster all logged failed queries by semantic similarity to detect systematic
    blind spots and produce targeted prompt adaptations and knowledge base actions.
    """
    from app.exemplars import cluster_failures
    clusters = cluster_failures()
    return {
        "status": "success",
        "total_clusters": len(clusters),
        "failure_clusters": clusters,
        "recommendation": (
            "Review failure clusters above. For critical knowledge gaps, add documents to db/documents/ "
            "and call POST /ingest. For reasoning errors, provide corrections via POST /feedback."
        ),
    }


@router.get("/feedback/stats", summary="Get aggregate feedback statistics")
def feedback_stats():
    """
    Return a summary of all collected feedback.

    Includes:
    - Total feedback count (good / bad breakdown)
    - Good/bad percentages
    - Average auto-evaluation score
    - Top 5 most queried topics
    """
    return analyze_feedback()


@router.get("/feedback/improvements", summary="Get queries and clusters that need improvement")
def feedback_improvements():
    """
    Surface queries that have received bad feedback multiple times, along with
    semantic failure cluster diagnostics.
    """
    hints = get_improvement_hints()
    retraining = retrain_hint()
    from app.exemplars import cluster_failures
    clusters = cluster_failures()

    return {
        "flagged_queries": hints,
        "total_flagged":   len(hints),
        "failure_clusters": clusters,
        "retraining_signal": retraining,
    }
