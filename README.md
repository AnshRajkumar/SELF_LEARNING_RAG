# 🧠 Self-Learning RAG System

A modular, production-ready **Retrieval-Augmented Generation (RAG)** backend built with FastAPI. It retrieves relevant documents using FAISS vector search, generates grounded answers via an LLM, and continuously improves through a feedback loop.

---

## 📁 Project Structure

```
selfimprovingrag/
├── app/
│   ├── __init__.py        # Python package marker
│   ├── main.py            # FastAPI entry point + middleware
│   ├── ingest.py          # Document loading & embedding
│   ├── retrieve.py        # FAISS vector search
│   ├── generate.py        # LLM answer generation
│   ├── ask.py             # Full RAG pipeline (retrieve → generate)
│   ├── evaluate.py        # Response quality scoring
│   ├── feedback.py        # Feedback storage & self-learning loop
│   └── logger.py          # Centralized logging
├── db/
│   ├── documents/
│   │   └── sample_docs.txt    # Sample knowledge base documents
│   ├── faiss.index            # [auto-generated] FAISS vector index
│   ├── metadata.json          # [auto-generated] Document metadata
│   ├── feedback_log.json      # [auto-generated] All feedback entries
│   ├── failed_queries.json    # [auto-generated] Bad-feedback queries
│   └── app.log                # [auto-generated] Application log
├── config.py              # Central configuration & settings
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run.

### 2. (Optional) Configure environment

```bash
copy .env.example .env
# Edit .env to set LLM_PROVIDER=openai and your OPENAI_API_KEY
```

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

The server will:
- Auto-ingest the sample documents
- Build the FAISS index
- Be ready at **http://127.0.0.1:8000**

### 4. Explore the API

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/ingest` | Re-ingest documents & rebuild dense/sparse indices |
| `POST` | `/retrieve` | Hybrid document retrieval (BM25 + FAISS + Re-ranking) |
| `POST` | `/generate` | LLM generation with adaptive reflection memory |
| `POST` | `/ask` | **Full RAG pipeline** with auto RAG Triad evaluation |
| `POST` | `/evaluate` | Multi-metric RAG Triad evaluation (Faithfulness, Relevancy, Precision) |
| `POST` | `/feedback` | Submit feedback with active learning corrections |
| `GET`  | `/feedback/exemplars` | View learned reflection memory & few-shot directives |
| `POST` | `/feedback/optimize` | Semantic failure clustering & root-cause diagnostics |
| `GET`  | `/feedback/stats` | Aggregate feedback statistics |
| `GET`  | `/feedback/improvements` | Queries and failure clusters needing improvement |

---

## 📡 Example Requests

### Health Check
```bash
curl http://127.0.0.1:8000/
```

---

### Re-ingest Documents
```bash
curl -X POST http://127.0.0.1:8000/ingest
```

---

### Retrieve Documents (Hybrid Search & Re-ranking)
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 3,
    "mode": "hybrid",
    "rerank": true
  }'
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "top_k": 3,
  "mode": "hybrid",
  "reranked": true,
  "results": [
    {
      "text": "Machine learning is a subset of artificial intelligence...",
      "source": "sample_docs.txt",
      "score": 1.0,
      "rank": 1,
      "channel": "hybrid_rrf"
    }
  ],
  "total_found": 3
}
```

---

### Generate Answer
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is FAISS?",
    "context": [
      {
        "text": "FAISS is a library for efficient similarity search...",
        "source": "sample_docs.txt"
      }
    ]
  }'
```

---

### Full RAG Pipeline (Main Endpoint)
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does retrieval-augmented generation work?", "top_k": 3}'
```

**Response:**
```json
{
  "query_id": "abc123-...",
  "query": "How does retrieval-augmented generation work?",
  "answer": "Based on the retrieved information...",
  "sources": [...],
  "auto_eval": {
    "score": 0.75,
    "label": "good",
    "details": "Keyword overlap score: 75%..."
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "pipeline": {
    "retrieve_ms": 12.3,
    "generate_ms": 2.1,
    "total_ms": 14.4,
    "docs_retrieved": 3
  }
}
```

---

### Submit Feedback
```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does retrieval-augmented generation work?",
    "context": [],
    "answer": "Based on the retrieved information...",
    "feedback": "good"
  }'
```

```bash
# Submit bad feedback with user correction (Triggers Active Self-Learning)
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum computing?",
    "context": [],
    "answer": "I could not find relevant information.",
    "feedback": "bad",
    "correction": "Quantum computing uses superposition and entanglement. State clearly that it is not covered in our documentation.",
    "failure_reason": "out_of_domain"
  }'
```

---

### Evaluate an Answer (RAG Triad)
```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "answer": "Deep learning uses neural networks with multiple layers.",
    "context": [
      {"text": "Deep learning is a subset of machine learning using multi-layered neural networks."}
    ]
  }'
```

**Response:**
```json
{
  "query": "What is deep learning?",
  "answer": "Deep learning uses neural networks with multiple layers.",
  "score": 1.0,
  "label": "good",
  "details": "RAG Triad [llm_judge]: Faithfulness=100.0%, Relevancy=100.0%, Precision=100.0%. Composite=100.0% -> GOOD.",
  "threshold": 0.5,
  "metrics": {
    "faithfulness": 1.0,
    "answer_relevancy": 1.0,
    "context_precision": 1.0
  },
  "engine": "llm_judge"
}
```

---

### Get Feedback Statistics
```bash
curl http://127.0.0.1:8000/feedback/stats
```

**Response:**
```json
{
  "total": 10,
  "good": 7,
  "bad": 3,
  "good_pct": 70.0,
  "bad_pct": 30.0,
  "avg_score": 0.62,
  "top_queries": [...]
}
```

---

### Get Improvement Hints
```bash
curl http://127.0.0.1:8000/feedback/improvements
```

**Response:**
```json
{
  "flagged_queries": [
    {
      "query": "What is quantum computing?",
      "failure_count": 3,
      "recommended_action": "Medium priority: Try adding targeted documents."
    }
  ],
  "total_flagged": 1,
  "retraining_signal": {
    "needs_retraining": false,
    "bad_feedback_pct": 30.0
  }
}
```

---

## 🔧 Configuration

Edit `config.py` or set values in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | `mock` or `groq` |
| `GROQ_API_KEY` | — | Required if `LLM_PROVIDER=groq`. Get free key at [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `DEFAULT_TOP_K` | `3` | Default retrieval count |
| `GOOD_SCORE_THRESHOLD` | `0.3` | Minimum score for "good" label |
| `BAD_FEEDBACK_ALERT_THRESHOLD` | `2` | Bad feedbacks before flagging |

---

## 🔮 Self-Learning Architecture

```
User Query
    │
    ▼
POST /ask ────────────────────────────────────────────────┐
    │                                                      │
    ├─ [1] RETRIEVE  → FAISS semantic search              │
    ├─ [2] GENERATE  → Mock LLM / OpenAI                  │
    ├─ [3] EVALUATE  → Keyword overlap scoring            │
    └─ Returns: answer + sources + auto_eval + query_id   │
                                                          │
User reviews answer                                       │
    │                                                      │
    ▼                                                      │
POST /feedback { feedback: "good" | "bad" }               │
    │                                                      │
    ├─ "good" → Stored in feedback_log.json               │
    └─ "bad"  → Also stored in failed_queries.json        │
                    │                                      │
                    ▼                                      │
           GET /feedback/improvements                      │
               Shows flagged queries + recommended actions │
               (re-ranking, prompt tuning, retraining)    │
```

### Future Improvement Hooks (pre-wired stubs)

| Hook | Location | Purpose |
|------|----------|---------|
| `re_rank()` | `retrieve.py` | Cross-encoder re-ranking |
| `tune_prompt()` | `generate.py` | Feedback-driven prompt adaptation |
| `retrain_hint()` | `feedback.py` | Signal when embeddings need retraining |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `sentence-transformers` | Text embeddings |
| `faiss-cpu` | Vector similarity search |
| `openai` | Optional OpenAI integration |
| `pydantic` | Request/response validation |
| `python-dotenv` | Environment variable loading |
| `numpy` | Array operations |

---

## 📝 Adding Your Own Documents

1. Add `.txt` files to `db/documents/`
2. Call `POST /ingest` to rebuild the index
3. Start querying with `POST /ask`

The system will automatically chunk your documents by paragraphs and embed them.

---

*Built with ❤️ for learning and scalable to production.*
