# ─────────────────────────────────────────────────────────────────────────────
# app/main.py  —  FastAPI application entry point
#
# This file:
#   1. Creates the FastAPI app with metadata (title, docs, version)
#   2. Registers all routers (ingest, retrieve, generate, ask, evaluate, feedback)
#   3. Adds logging middleware to trace every HTTP request
#   4. Runs the ingestion pipeline on startup so the server is ready immediately
#   5. Exposes a health-check endpoint at GET /
#
# Start the server with:
#   uvicorn app.main:app --reload
# ─────────────────────────────────────────────────────────────────────────────

import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow config.py to be found at the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLM_PROVIDER, EMBEDDING_MODEL
from app.logger import logger

# Import all routers
from app import ingest, retrieve, generate, ask, evaluate, feedback


# ── Lifespan: startup + shutdown logic ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code inside the 'async with' block runs at startup.
    Code after 'yield' runs at shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Self-Learning RAG System — Starting up")
    logger.info("=" * 60)
    logger.info(f"  Embedding model : {EMBEDDING_MODEL}")
    logger.info(f"  LLM provider    : {LLM_PROVIDER}")
    logger.info("=" * 60)

    # Initialize SQLite database (WAL mode + automatic JSON migration)
    try:
        from app.db import init_db
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Auto-ingest sample documents so the system is ready on first request
    logger.info("Running initial document ingestion...")
    try:
        result = ingest.run_ingestion()
        logger.info(f"Ingestion result: {result}")
    except Exception as e:
        logger.error(f"Ingestion failed at startup: {e}")
        logger.warning("Server will start, but /retrieve will fail until /ingest is called.")

    # Pre-load the FAISS index, metadata, and BM25 index into memory (warm-up)
    try:
        retrieve.load_index()
        retrieve.load_metadata()
        retrieve.load_bm25()
        logger.info("FAISS index, metadata, and BM25 pre-loaded into memory.")
    except Exception as e:
        logger.warning(f"Index pre-load skipped: {e}")

    logger.info("Server is ready! Visit http://127.0.0.1:8000/docs for the API explorer.")

    yield  # Server runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Self-Learning RAG System shutting down. Goodbye!")


# ── FastAPI application ────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Learning RAG System",
    description=(
        "A modular Retrieval-Augmented Generation (RAG) backend with a self-learning "
        "feedback loop. Ask questions, get grounded answers, submit feedback, and watch "
        "the system track what needs improvement.\n\n"
        "**Quick start:**\n"
        "1. `POST /ask` — Ask a question (full RAG pipeline)\n"
        "2. `POST /feedback` — Rate the answer (good / bad)\n"
        "3. `GET /feedback/improvements` — See what needs fixing\n"
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
    lifespan=lifespan,
)


# ── CORS Middleware ────────────────────────────────────────────────────────
# Allow all origins for development. Restrict in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming HTTP request with method, path, and response time.
    This runs for every request before and after the endpoint handler.
    """
    start_time = time.perf_counter()

    # Process the request
    response = await call_next(request)

    # Calculate how long the request took
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} "
        f"[{duration_ms:.1f}ms]"
    )

    return response


# ── Register routers ───────────────────────────────────────────────────────
# Each router handles a specific set of endpoints

app.include_router(ingest.router,   tags=["Ingestion"])
app.include_router(retrieve.router, tags=["Retrieval"])
app.include_router(generate.router, tags=["Generation"])
app.include_router(ask.router,      tags=["RAG Pipeline"])
app.include_router(evaluate.router, tags=["Evaluation"])
app.include_router(feedback.router, tags=["Feedback & Self-Learning"])


@app.get("/db/status", tags=["Health"], summary="SQLite Database Health & Metrics")
def db_status():
    """
    Returns database storage metrics, table row counts, and SQLite WAL status.
    """
    from app.db import get_db
    from config import SQLITE_DB_PATH
    with get_db() as conn:
        q_count = conn.execute("SELECT COUNT(*) FROM queries;").fetchone()[0]
        ev_count = conn.execute("SELECT COUNT(*) FROM evaluations;").fetchone()[0]
        fb_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
        fl_count = conn.execute("SELECT COUNT(*) FROM failed_queries;").fetchone()[0]
        ex_count = conn.execute("SELECT COUNT(*) FROM exemplars;").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

    size_kb = round(SQLITE_DB_PATH.stat().st_size / 1024, 2) if SQLITE_DB_PATH.exists() else 0
    return {
        "status": "healthy",
        "database": "SQLite",
        "journal_mode": journal_mode,
        "db_size_kb": size_kb,
        "table_counts": {
            "queries": q_count,
            "evaluations": ev_count,
            "feedback": fb_count,
            "failed_queries": fl_count,
            "exemplars": ex_count,
        },
    }


# ── Health check endpoint ──────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Health check")
def root():
    """
    Server health check. Returns basic system info.
    Useful for checking if the server is running and which mode it's in.
    """
    return {
        "status":          "ok",
        "system":          "Self-Learning RAG",
        "version":         "1.0.0",
        "llm_provider":    LLM_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "docs":            "http://127.0.0.1:8000/docs",
        "endpoints": {
            "ingest":      "POST /ingest",
            "retrieve":    "POST /retrieve",
            "generate":    "POST /generate",
            "ask":         "POST /ask",
            "evaluate":    "POST /evaluate",
            "feedback":    "POST /feedback",
            "stats":       "GET /feedback/stats",
            "improvements":"GET /feedback/improvements",
        },
    }


# ── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exceptions and return a clean JSON error response.
    Always logs the full error for debugging.
    """
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal server error",
            "detail":  str(exc),
            "path":    str(request.url.path),
        },
    )
