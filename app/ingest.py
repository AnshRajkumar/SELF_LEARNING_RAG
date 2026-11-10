# ─────────────────────────────────────────────────────────────────────────────
# app/ingest.py  —  Document loading, embedding, and FAISS index construction
#
# Flow:
#   1. Load raw .txt files from db/documents/
#   2. Split them into individual paragraphs
#   3. Encode each paragraph with sentence-transformers
#   4. Build a FAISS index and save it to disk
#   5. Save doc texts + metadata to metadata.json for lookup later
#
# Endpoint: POST /ingest
# ─────────────────────────────────────────────────────────────────────────────

import json
import sys
import numpy as np
from pathlib import Path
from fastapi import APIRouter

# Allow imports from project root (config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DOCS_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL,
)
from app.logger import logger

# FastAPI router — registered in main.py
router = APIRouter()

# ── Globals (shared state within this module) ──────────────────────────────
# Loaded once at startup and reused for every embed call
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazily load the embedding model.
    The model is downloaded on first call (~80 MB) and cached to disk.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _model


def load_documents() -> list[dict]:
    """
    Read all .txt files from the documents directory.
    Each non-empty paragraph (split by blank line) becomes one document chunk.

    Returns:
        List of dicts: [{"text": "...", "source": "filename.txt"}, ...]
    """
    documents = []

    # Make sure the directory exists
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {DOCS_DIR}. Nothing to ingest.")
        return documents

    for filepath in txt_files:
        logger.info(f"Loading document: {filepath.name}")
        content = filepath.read_text(encoding="utf-8")

        # Split on blank lines to get individual paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for para in paragraphs:
            documents.append({
                "text": para,
                "source": filepath.name,
            })

    logger.info(f"Loaded {len(documents)} document chunks from {len(txt_files)} file(s).")
    return documents


def embed_documents(documents: list[dict]) -> np.ndarray:
    """
    Convert document texts into embedding vectors using sentence-transformers.

    Args:
        documents: List of document dicts with a "text" key.

    Returns:
        numpy array of shape (num_docs, embedding_dim) as float32.
    """
    model = get_model()
    texts = [doc["text"] for doc in documents]

    logger.info(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity with FAISS
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat index (exact search) from embeddings.
    IndexFlatIP performs inner-product (cosine) search since we L2-normalized.

    Args:
        embeddings: float32 numpy array (num_docs, dim)

    Returns:
        FAISS index populated with all embeddings.
    """
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension={dim}, vectors={len(embeddings)}")

    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (after normalization)
    index.add(embeddings)

    logger.info(f"FAISS index built. Total vectors: {index.ntotal}")
    return index


def save_index_and_metadata(index: faiss.Index, documents: list[dict]) -> None:
    """
    Persist the FAISS index and document metadata to disk.

    Args:
        index: Populated FAISS index.
        documents: Original list of document dicts.
    """
    # Ensure db/ directory exists
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index binary
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # Save document metadata (text + source) as JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {METADATA_PATH}")


def run_ingestion() -> dict:
    """
    Full ingestion pipeline: load → embed → index → save.
    Called at server startup and via POST /ingest.

    Returns:
        Summary dict with counts.
    """
    logger.info("=== Starting ingestion pipeline ===")

    documents = load_documents()
    if not documents:
        return {"status": "no_documents", "chunks": 0}

    embeddings = embed_documents(documents)
    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, documents)

    logger.info("=== Ingestion pipeline complete ===")
    return {
        "status": "success",
        "chunks": len(documents),
        "embedding_dim": int(embeddings.shape[1]),
    }


# ── API Endpoint ───────────────────────────────────────────────────────────

@router.post("/ingest", summary="Re-ingest all documents and rebuild indices")
def ingest_documents():
    """
    Trigger document ingestion on demand.
    Rebuilds the FAISS dense vector index and refreshes BM25 and metadata caches.

    Returns:
        JSON with ingestion status and number of chunks indexed.
    """
    result = run_ingestion()
    try:
        from app.retrieve import reload_index
        reload_index()
    except Exception as e:
        logger.warning(f"Index reload after ingestion warning: {e}")
    return result
