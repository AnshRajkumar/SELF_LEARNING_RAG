# ─────────────────────────────────────────────────────────────────────────────
# config.py  —  Central configuration for the Self-Learning RAG system
#
# All tuneable parameters live here so you never have to hunt through code.
# Secrets (API keys) are loaded from a .env file via python-dotenv.
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (won't fail if it doesn't)
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent          # project root
DB_DIR          = BASE_DIR / "db"                # all persistence goes here
DOCS_DIR        = DB_DIR / "documents"           # raw text documents
FAISS_INDEX_PATH = DB_DIR / "faiss.index"        # saved FAISS index file
METADATA_PATH   = DB_DIR / "metadata.json"       # doc text + source mapping
FEEDBACK_LOG    = DB_DIR / "feedback_log.json"   # every feedback entry
FAILED_QUERIES  = DB_DIR / "failed_queries.json" # queries with bad feedback
EXEMPLARS_PATH  = DB_DIR / "exemplars.json"      # learned self-improvement memory
SQLITE_DB_PATH  = DB_DIR / "rag_store.db"        # SQLite structured store (ACID + WAL mode)
LOG_FILE        = DB_DIR / "app.log"             # application log

# ── Embedding model ────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is small (80 MB), fast, and very accurate for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Retrieval & Re-ranking settings ──────────────────────────────────────
DEFAULT_TOP_K       = int(os.getenv("DEFAULT_TOP_K", "3"))    # documents returned per query
MIN_SCORE           = float(os.getenv("MIN_SCORE", "0.0"))    # score threshold filter

# Hybrid Search (FAISS dense + BM25 sparse)
RETRIEVAL_MODE      = os.getenv("RETRIEVAL_MODE", "hybrid")   # "hybrid", "dense", "sparse"
RRF_K               = int(os.getenv("RRF_K", "60"))           # Reciprocal Rank Fusion constant
DENSE_WEIGHT        = float(os.getenv("DENSE_WEIGHT", "0.6")) # Weight for dense vector search
SPARSE_WEIGHT       = float(os.getenv("SPARSE_WEIGHT", "0.4"))# Weight for BM25 keyword search

# Re-ranking (Cross-Encoder / LLM judge)
RERANKER_ENABLED    = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1")
RERANKER_MODEL      = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CANDIDATE_FACTOR    = int(os.getenv("CANDIDATE_FACTOR", "2")) # Candidates to pull before reranking (top_k * factor)

# ── LLM settings ──────────────────────────────────────────────────────────
# "mock" → uses built-in template-based response (no API key needed)
# "groq"  → calls Groq ultra-fast inference API (requires GROQ_API_KEY in .env)
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "mock")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile = best quality | llama-3.1-8b-instant = fastest
GROQ_MODEL      = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# ── Evaluation settings (RAG Triad / Ragas Metrics) ────────────────────────
# Minimum composite score for "good" quality label
GOOD_SCORE_THRESHOLD      = float(os.getenv("GOOD_SCORE_THRESHOLD", "0.50"))
# Minimum faithfulness score to avoid hallucination flagging
FAITHFULNESS_THRESHOLD    = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.60"))
# RAG Triad composite weights (sum to 1.0)
WEIGHT_FAITHFULNESS       = float(os.getenv("WEIGHT_FAITHFULNESS", "0.40"))
WEIGHT_ANSWER_RELEVANCY   = float(os.getenv("WEIGHT_ANSWER_RELEVANCY", "0.35"))
WEIGHT_CONTEXT_PRECISION  = float(os.getenv("WEIGHT_CONTEXT_PRECISION", "0.25"))

# ── Self-learning settings ─────────────────────────────────────────────────
# How many bad feedbacks for the same query before we flag it for improvement
BAD_FEEDBACK_ALERT_THRESHOLD    = int(os.getenv("BAD_FEEDBACK_ALERT_THRESHOLD", "2"))
# Similarity threshold to match incoming queries against learned failure exemplars
EXEMPLAR_SIMILARITY_THRESHOLD   = float(os.getenv("EXEMPLAR_SIMILARITY_THRESHOLD", "0.60"))
# Maximum number of learned guidance exemplars to inject into generation prompt
MAX_EXEMPLARS_IN_PROMPT         = int(os.getenv("MAX_EXEMPLARS_IN_PROMPT", "2"))
# Automatically widen search window and expand query on problematic query topics
ADAPTIVE_RETRIEVAL_ENABLED      = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").lower() in ("true", "1")
