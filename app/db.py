# ─────────────────────────────────────────────────────────────────────────────
# app/db.py  —  Structured SQLite Database Layer (ACID + WAL mode)
#
# Replaces fragile flat JSON files with a high-performance relational database:
#   - Queries: Trace every RAG pipeline request, latency, and retrieved chunk count
#   - Evaluations: Store RAG Triad scores (Faithfulness, Relevancy, Precision)
#   - Feedback: Structured good/bad ratings, user corrections, failure reasons
#   - Failed Queries: Failure frequency tracking and priority indexing
#   - Exemplars: Episodic memory store for learned few-shot prompt guidelines
#
# Automatically migrates existing legacy JSON files into SQLite on initialization.
# ─────────────────────────────────────────────────────────────────────────────

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SQLITE_DB_PATH,
    FEEDBACK_LOG,
    FAILED_QUERIES,
    EXEMPLARS_PATH,
    BAD_FEEDBACK_ALERT_THRESHOLD,
)
from app.logger import logger


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Thread-safe context manager for SQLite database operations.
    Enables WAL mode (Write-Ahead Logging) for concurrent reads and writes.
    """
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize database schema with tables, constraints, and indexes.
    Automatically runs migration from legacy JSON files if database is empty.
    """
    logger.info(f"Initializing SQLite database at: {SQLITE_DB_PATH}")

    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                retrieve_ms REAL,
                generate_ms REAL,
                total_ms REAL,
                docs_retrieved INTEGER
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                query_text TEXT NOT NULL,
                composite_score REAL NOT NULL,
                faithfulness REAL,
                answer_relevancy REAL,
                context_precision REAL,
                label TEXT NOT NULL,
                engine TEXT NOT NULL,
                details TEXT,
                critique TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries(query_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                query_id TEXT,
                query_text TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('good', 'bad')),
                auto_score REAL,
                correction TEXT,
                failure_reason TEXT,
                sources_json TEXT,
                context_snippet TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries(query_id)
            );

            CREATE TABLE IF NOT EXISTS failed_queries (
                query_normalized TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                failure_count INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                latest_answer TEXT,
                latest_score REAL
            );

            CREATE TABLE IF NOT EXISTS exemplars (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT,
                correction TEXT NOT NULL,
                failure_reason TEXT,
                context_summary TEXT,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(timestamp);
            CREATE INDEX IF NOT EXISTS idx_eval_query ON evaluations(query_id);
            CREATE INDEX IF NOT EXISTS idx_fb_rating ON feedback(rating);
            CREATE INDEX IF NOT EXISTS idx_failed_count ON failed_queries(failure_count DESC);
        """)

    # Migrate existing data from JSON files if database was just created
    _migrate_legacy_json_data()


def _migrate_legacy_json_data() -> None:
    """Safely migrate existing JSON logs into SQLite if tables are empty."""
    with get_db() as conn:
        fb_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
        if fb_count == 0 and FEEDBACK_LOG.exists():
            try:
                with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for item in entries:
                    conn.execute("""
                        INSERT OR IGNORE INTO feedback (
                            id, query_id, query_text, answer_text, rating,
                            auto_score, correction, failure_reason, sources_json,
                            context_snippet, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        item.get("id"),
                        item.get("query_id"),
                        item.get("query", ""),
                        item.get("answer", ""),
                        item.get("feedback", "good"),
                        item.get("auto_score"),
                        item.get("correction"),
                        item.get("failure_reason"),
                        json.dumps(item.get("sources", [])),
                        item.get("context_snippet", ""),
                        item.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                    ))
                logger.info(f"Migrated {len(entries)} legacy feedback records into SQLite.")
            except Exception as e:
                logger.warning(f"Legacy feedback migration skipped: {e}")

        failed_count = conn.execute("SELECT COUNT(*) FROM failed_queries;").fetchone()[0]
        if failed_count == 0 and FAILED_QUERIES.exists():
            try:
                with open(FAILED_QUERIES, "r", encoding="utf-8") as f:
                    failed = json.load(f)
                for item in failed:
                    q = item.get("query", "").strip()
                    if not q:
                        continue
                    conn.execute("""
                        INSERT OR IGNORE INTO failed_queries (
                            query_normalized, query_text, failure_count,
                            first_seen, last_seen, latest_answer, latest_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """, (
                        q.lower(),
                        q,
                        item.get("failure_count", 1),
                        item.get("first_seen", datetime.utcnow().isoformat() + "Z"),
                        item.get("last_seen", datetime.utcnow().isoformat() + "Z"),
                        item.get("latest_answer", ""),
                        item.get("latest_score"),
                    ))
                logger.info(f"Migrated {len(failed)} legacy failed queries into SQLite.")
            except Exception as e:
                logger.warning(f"Legacy failed query migration skipped: {e}")

        ex_count = conn.execute("SELECT COUNT(*) FROM exemplars;").fetchone()[0]
        if ex_count == 0 and EXEMPLARS_PATH.exists():
            try:
                with open(EXEMPLARS_PATH, "r", encoding="utf-8") as f:
                    exemplars = json.load(f)
                for item in exemplars:
                    conn.execute("""
                        INSERT OR IGNORE INTO exemplars (
                            id, query, answer, correction, failure_reason,
                            context_summary, embedding_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        item.get("id"),
                        item.get("query", ""),
                        item.get("answer", ""),
                        item.get("correction", ""),
                        item.get("failure_reason", ""),
                        item.get("context_summary", ""),
                        json.dumps(item.get("embedding", [])),
                        item.get("created_at", datetime.utcnow().isoformat() + "Z"),
                    ))
                logger.info(f"Migrated {len(exemplars)} legacy exemplars into SQLite.")
            except Exception as e:
                logger.warning(f"Legacy exemplars migration skipped: {e}")


# ── Database Operations: Pipeline Traces ───────────────────────────────────

def log_pipeline_run(
    query_id: str,
    query: str,
    pipeline_meta: dict,
    auto_eval: dict,
) -> None:
    """Store complete trace of a RAG query run including latency and quality scores."""
    ts = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO queries (
                query_id, query_text, timestamp, retrieve_ms, generate_ms, total_ms, docs_retrieved
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            query_id,
            query,
            ts,
            pipeline_meta.get("retrieve_ms", 0.0),
            pipeline_meta.get("generate_ms", 0.0),
            pipeline_meta.get("total_ms", 0.0),
            pipeline_meta.get("docs_retrieved", 0),
        ))

        metrics = auto_eval.get("metrics", {})
        conn.execute("""
            INSERT INTO evaluations (
                query_id, query_text, composite_score, faithfulness,
                answer_relevancy, context_precision, label, engine,
                details, critique, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            query_id,
            query,
            auto_eval.get("score", 0.0),
            metrics.get("faithfulness"),
            metrics.get("answer_relevancy"),
            metrics.get("context_precision"),
            auto_eval.get("label", "unknown"),
            auto_eval.get("engine", "unknown"),
            auto_eval.get("details", ""),
            auto_eval.get("critique", ""),
            ts,
        ))


# ── Database Operations: Feedback & Self-Learning ──────────────────────────

def insert_feedback(
    entry_id: str,
    query: str,
    answer: str,
    rating: str,
    auto_score: Optional[float] = None,
    correction: Optional[str] = None,
    failure_reason: Optional[str] = None,
    sources: Optional[list] = None,
    context_snippet: Optional[str] = None,
    query_id: Optional[str] = None,
) -> dict:
    """Record user feedback and update failed_queries tracking."""
    ts = datetime.utcnow().isoformat() + "Z"

    with get_db() as conn:
        conn.execute("""
            INSERT INTO feedback (
                id, query_id, query_text, answer_text, rating,
                auto_score, correction, failure_reason, sources_json,
                context_snippet, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            entry_id,
            query_id,
            query,
            answer,
            rating,
            auto_score,
            correction,
            failure_reason,
            json.dumps(sources or []),
            context_snippet or "",
            ts,
        ))

        if rating == "bad":
            norm_q = query.strip().lower()
            existing = conn.execute(
                "SELECT failure_count, first_seen FROM failed_queries WHERE query_normalized = ?;",
                (norm_q,)
            ).fetchone()

            if existing:
                new_count = existing["failure_count"] + 1
                conn.execute("""
                    UPDATE failed_queries
                    SET failure_count = ?, last_seen = ?, latest_answer = ?, latest_score = ?
                    WHERE query_normalized = ?;
                """, (new_count, ts, answer, auto_score, norm_q))
            else:
                conn.execute("""
                    INSERT INTO failed_queries (
                        query_normalized, query_text, failure_count,
                        first_seen, last_seen, latest_answer, latest_score
                    ) VALUES (?, ?, 1, ?, ?, ?, ?);
                """, (norm_q, query, ts, ts, answer, auto_score))

    return {
        "id": entry_id,
        "query": query,
        "feedback": rating,
        "timestamp": ts,
    }


def get_feedback_stats_db() -> dict:
    """Compute aggregate feedback statistics via SQL queries."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN rating = 'good' THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN rating = 'bad' THEN 1 ELSE 0 END) as bad,
                AVG(auto_score) as avg_score
            FROM feedback;
        """).fetchone()

        total = row["total"] or 0
        good = row["good"] or 0
        bad = row["bad"] or 0
        avg_score = round(float(row["avg_score"]), 4) if row["avg_score"] is not None else None

        # Top 5 most queried topics from feedback
        top_rows = conn.execute("""
            SELECT query_text, COUNT(*) as count
            FROM feedback
            GROUP BY query_text
            ORDER BY count DESC
            LIMIT 5;
        """).fetchall()

        top_queries = [{"query": r["query_text"], "count": r["count"]} for r in top_rows]

    return {
        "total": total,
        "good": good,
        "bad": bad,
        "good_pct": round(good / total * 100, 1) if total > 0 else 0.0,
        "bad_pct": round(bad / total * 100, 1) if total > 0 else 0.0,
        "avg_score": avg_score,
        "top_queries": top_queries,
    }


def get_all_failed_queries_db() -> list[dict]:
    """Retrieve all failed queries sorted by failure frequency."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT query_text as query, failure_count, first_seen, last_seen, latest_answer, latest_score
            FROM failed_queries
            ORDER BY failure_count DESC;
        """).fetchall()
        return [dict(r) for r in rows]


def get_flagged_queries_db(threshold: int = BAD_FEEDBACK_ALERT_THRESHOLD) -> list[dict]:
    """Return queries whose failure count meets or exceeds the alert threshold."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT query_text as query, failure_count, first_seen, last_seen
            FROM failed_queries
            WHERE failure_count >= ?
            ORDER BY failure_count DESC;
        """, (threshold,)).fetchall()

    hints = []
    for r in rows:
        count = r["failure_count"]
        if count >= 5:
            action = "High priority: Consider adding targeted documents or retraining embeddings."
        elif count >= 3:
            action = "Medium priority: Try prompt engineering or adding targeted documents."
        else:
            action = "Low priority: Monitor for additional failures before taking action."

        hints.append({
            "query": r["query"],
            "failure_count": count,
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "recommended_action": action,
        })
    return hints


# ── Database Operations: Exemplars Store ───────────────────────────────────

def get_all_exemplars_db() -> list[dict]:
    """Retrieve all learned reflection exemplars with parsed embeddings."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, query, answer, correction, failure_reason, context_summary, embedding_json, created_at
            FROM exemplars
            ORDER BY created_at ASC;
        """).fetchall()

    exemplars = []
    for r in rows:
        item = dict(r)
        item["embedding"] = json.loads(item.pop("embedding_json", "[]"))
        exemplars.append(item)
    return exemplars


def upsert_exemplar_db(
    record_id: str,
    query: str,
    answer: str,
    correction: str,
    failure_reason: str,
    context_summary: str,
    embedding: list[float],
) -> dict:
    """Insert or update a learned corrective exemplar in SQLite."""
    ts = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        # Check if an exemplar for this query already exists
        existing = conn.execute(
            "SELECT id FROM exemplars WHERE LOWER(query) = LOWER(?);",
            (query.strip(),)
        ).fetchone()

        if existing:
            rec_id = existing["id"]
            conn.execute("""
                UPDATE exemplars
                SET correction = ?, failure_reason = ?, context_summary = ?,
                    embedding_json = ?, answer = ?
                WHERE id = ?;
            """, (
                correction,
                failure_reason,
                context_summary,
                json.dumps(embedding),
                answer,
                rec_id,
            ))
        else:
            rec_id = record_id
            conn.execute("""
                INSERT INTO exemplars (
                    id, query, answer, correction, failure_reason,
                    context_summary, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                rec_id,
                query,
                answer,
                correction,
                failure_reason,
                context_summary,
                json.dumps(embedding),
                ts,
            ))

    return {
        "id": rec_id,
        "query": query,
        "correction": correction,
        "failure_reason": failure_reason,
        "created_at": ts,
    }


# Auto-initialize database schema on module load
try:
    init_db()
except Exception as _e:
    logger.warning(f"Database auto-init deferred: {_e}")
