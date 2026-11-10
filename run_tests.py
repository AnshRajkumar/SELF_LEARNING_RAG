# ─────────────────────────────────────────────────────────────────────────────
# run_tests.py  —  In-process end-to-end test suite using FastAPI TestClient
#
# Tests the full Self-Learning RAG pipeline including the newly implemented
# RAG Triad evaluation metrics (Faithfulness, Relevancy, Context Precision).
# ─────────────────────────────────────────────────────────────────────────────

import json
from fastapi.testclient import TestClient
from app.db import init_db
from app.main import app

# Ensure database schema and migrations are ready
init_db()

client = TestClient(app)

print("=" * 65)
print("  SELF-LEARNING RAG — AUTOMATED TEST SUITE (RAG TRIAD EVALUATION)")
print("=" * 65)

# TEST 1: Health check
print("\n[TEST 1] GET / — Health Check")
res = client.get("/")
assert res.status_code == 200, res.text
h = res.json()
print(f"  status       : {h['status']}")
print(f"  llm_provider : {h['llm_provider']}")
print(f"  embed_model  : {h['embedding_model']}")
assert h["status"] == "ok"
print("  PASS")

# TEST 1b: Database health & migration check
print("\n[TEST 1b] GET /db/status — SQLite Relational Database Health")
res = client.get("/db/status")
assert res.status_code == 200, res.text
db_stat = res.json()
print(f"  engine       : {db_stat['database']}")
print(f"  journal mode : {db_stat['journal_mode']}")
print(f"  db size (KB) : {db_stat['db_size_kb']}")
print(f"  table counts : {db_stat['table_counts']}")
assert db_stat["status"] == "healthy"
assert db_stat["journal_mode"].lower() == "wal"
print("  PASS")

# TEST 2a: Hybrid retrieval (BM25 + FAISS with RRF and Re-ranking)
print("\n[TEST 2a] POST /retrieve — Hybrid Search (BM25 + FAISS + RRF)")
res = client.post("/retrieve", json={"query": "What is machine learning?", "top_k": 2, "mode": "hybrid"})
assert res.status_code == 200, res.text
r = res.json()
print(f"  mode         : {r['mode']}")
print(f"  reranked     : {r['reranked']}")
print(f"  docs found   : {r['total_found']}")
for d in r["results"]:
    print(f"  [{d['rank']}] score={d['score']} | channel={d.get('channel')} | {d['text'][:60]}...")
assert r["total_found"] > 0
print("  PASS")

# TEST 2b: Dense-only retrieval
print("\n[TEST 2b] POST /retrieve — Dense-only Vector Search (FAISS)")
res = client.post("/retrieve", json={"query": "neural network architecture", "top_k": 2, "mode": "dense", "rerank": False})
assert res.status_code == 200, res.text
r_dense = res.json()
print(f"  mode         : {r_dense['mode']}")
print(f"  top match    : {r_dense['results'][0]['text'][:60]}...")
assert r_dense["total_found"] > 0
print("  PASS")

# TEST 2c: Sparse-only retrieval (BM25 keyword matching)
print("\n[TEST 2c] POST /retrieve — Sparse-only Lexical Search (BM25)")
res = client.post("/retrieve", json={"query": "FAISS similarity search", "top_k": 2, "mode": "sparse", "rerank": False})
assert res.status_code == 200, res.text
r_sparse = res.json()
print(f"  mode         : {r_sparse['mode']}")
print(f"  top match    : {r_sparse['results'][0]['text'][:60]}...")
assert r_sparse["total_found"] > 0
print("  PASS")

# TEST 3: Full RAG pipeline (POST /ask) with automatic RAG Triad evaluation
print("\n[TEST 3] POST /ask — Full RAG Pipeline & Auto-Evaluation")
res = client.post("/ask", json={"query": "What is retrieval-augmented generation?", "top_k": 3})
assert res.status_code == 200, res.text
a = res.json()
print(f"  query_id     : {a['query_id']}")
print(f"  docs_used    : {a['pipeline']['docs_retrieved']}")
print(f"  retrieve_ms  : {a['pipeline']['retrieve_ms']}ms")
print(f"  generate_ms  : {a['pipeline']['generate_ms']}ms")
print(f"  answer       : {a['answer'][:180]}...")
print(f"  quality      : {a['auto_eval']['label']} (composite={a['auto_eval']['score']})")
print(f"  eval details : {a['auto_eval']['details']}")
assert a["answer"]
assert "score" in a["auto_eval"]
assert "metrics" in a["auto_eval"]
print("  PASS")

# TEST 4a: Standalone evaluation (without context)
print("\n[TEST 4a] POST /evaluate — Standalone Answer Relevancy")
res = client.post("/evaluate", json={
    "query": "What is retrieval-augmented generation?",
    "answer": a["answer"]
})
assert res.status_code == 200, res.text
e1 = res.json()
print(f"  engine       : {e1['engine']}")
print(f"  score        : {e1['score']}")
print(f"  label        : {e1['label']}")
print(f"  details      : {e1['details']}")
assert "score" in e1
print("  PASS")

# TEST 4b: Full RAG Triad evaluation (with retrieved context)
print("\n[TEST 4b] POST /evaluate — Full RAG Triad (Faithfulness, Relevancy, Precision)")
res = client.post("/evaluate", json={
    "query": a["query"],
    "answer": a["answer"],
    "context": a["sources"]
})
assert res.status_code == 200, res.text
e2 = res.json()
print(f"  engine            : {e2['engine']}")
print(f"  composite score   : {e2['score']}")
print(f"  faithfulness      : {e2['metrics']['faithfulness']}")
print(f"  answer_relevancy  : {e2['metrics']['answer_relevancy']}")
print(f"  context_precision : {e2['metrics']['context_precision']}")
print(f"  details           : {e2['details']}")
assert e2["metrics"]["faithfulness"] is not None
assert e2["metrics"]["context_precision"] is not None
print("  PASS")

# TEST 4c: Hallucination detection test
print("\n[TEST 4c] POST /evaluate — Hallucination / Ungrounded Claim Detection")
res = client.post("/evaluate", json={
    "query": "What is retrieval-augmented generation?",
    "answer": "Retrieval-augmented generation is built on quantum entangled superconducting qubits operating at 4 Kelvin.",
    "context": a["sources"]
})
assert res.status_code == 200, res.text
e3 = res.json()
print(f"  label             : {e3['label']}")
print(f"  composite score   : {e3['score']}")
print(f"  faithfulness      : {e3['metrics']['faithfulness']}")
print(f"  details           : {e3['details']}")
# A completely fabricated quantum claim must be flagged as bad / low faithfulness!
assert e3["label"] == "bad" or (e3["metrics"]["faithfulness"] is not None and e3["metrics"]["faithfulness"] < 0.5)
print("  PASS — Hallucination successfully detected and rejected!")

# TEST 5: Submit good feedback
print("\n[TEST 5] POST /feedback — Submit GOOD feedback")
res = client.post("/feedback", json={
    "query": a["query"],
    "context": a["sources"],
    "answer": a["answer"],
    "feedback": "good",
    "auto_score": a["auto_eval"]["score"]
})
assert res.status_code == 200, res.text
f1 = res.json()
print(f"  id           : {f1['id']}")
print(f"  message      : {f1['message']}")
assert f1["status"] == "success"
print("  PASS")

# TEST 6a: Submit bad feedback with active learning correction
print("\n[TEST 6a] POST /feedback — Submit BAD feedback WITH User Correction (Active Learning)")
res = client.post("/feedback", json={
    "query": "What is quantum computing?",
    "context": [],
    "answer": "I could not find relevant information in the documents.",
    "feedback": "bad",
    "auto_score": 0.1,
    "correction": "Quantum computing uses superposition and entanglement. State that it is outside our internal documentation.",
    "failure_reason": "out_of_domain"
})
assert res.status_code == 200, res.text
f2 = res.json()
print(f"  id                : {f2['id']}")
print(f"  message           : {f2['message']}")
print(f"  exemplar_recorded : {f2['exemplar_recorded']}")
assert f2["status"] == "success"
assert f2["exemplar_recorded"] is True
print("  PASS — Corrective exemplar learned!")

# TEST 6b: Verify exemplar is active in episodic memory
print("\n[TEST 6b] GET /feedback/exemplars — Inspect Learned Episodic Memory")
res = client.get("/feedback/exemplars")
assert res.status_code == 200, res.text
ex_data = res.json()
print(f"  total exemplars   : {ex_data['total_exemplars']}")
assert ex_data["total_exemplars"] > 0
print(f"  latest exemplar   : Q='{ex_data['exemplars'][-1]['query']}' -> Guideline='{ex_data['exemplars'][-1]['correction'][:50]}...'")
print("  PASS")

# TEST 6c: Test Dynamic In-Context Adaptation on a new, semantically similar query
print("\n[TEST 6c] POST /generate — Verify Runtime Adaptation on Semantically Similar Query")
# Query is phrased differently: "Explain quantum computers" vs learned "What is quantum computing?"
res = client.post("/generate", json={
    "query": "Explain quantum computers",
    "context": [],
    "use_exemplars": True
})
assert res.status_code == 200, res.text
gen_res = res.json()
print(f"  query             : {gen_res['query']}")
print(f"  exemplars used    : {gen_res['exemplars_used']}")
print(f"  answer            : {gen_res['answer'][:180]}...")
assert gen_res["exemplars_used"] > 0
print("  PASS — Prompt successfully adapted using learned episodic memory!")

# TEST 6d: Test semantic failure clustering optimization
print("\n[TEST 6d] POST /feedback/optimize — Semantic Failure Topic Clustering")
res = client.post("/feedback/optimize")
assert res.status_code == 200, res.text
opt = res.json()
print(f"  status            : {opt['status']}")
print(f"  total clusters    : {opt['total_clusters']}")
for cl in opt["failure_clusters"]:
    print(f"  Cluster '{cl['theme']}': {cl['total_failures']} failures | Action: {cl['recommended_action']}")
assert opt["status"] == "success"
print("  PASS")

# TEST 7: Aggregate feedback statistics
print("\n[TEST 7] GET /feedback/stats — Feedback Statistics")
res = client.get("/feedback/stats")
assert res.status_code == 200, res.text
s = res.json()
print(f"  total        : {s['total']}")
print(f"  good         : {s['good']} ({s['good_pct']}%)")
print(f"  bad          : {s['bad']} ({s['bad_pct']}%)")
print(f"  avg_score    : {s['avg_score']}")
assert s["total"] >= 2
print("  PASS")

# TEST 8: Improvement recommendations with cluster diagnostics
print("\n[TEST 8] GET /feedback/improvements — Flagged Queries & Actionable Hints")
res = client.get("/feedback/improvements")
assert res.status_code == 200, res.text
imp = res.json()
print(f"  total flagged     : {imp['total_flagged']}")
print(f"  failure clusters  : {len(imp.get('failure_clusters', []))}")
for q in imp["flagged_queries"]:
    print(f"  -> '{q['query']}' (failures: {q['failure_count']}): {q['recommended_action']}")
print("  PASS")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED! RAG TRIAD & SELF-LEARNING ARE FULLY OPERATIONAL.")
print("=" * 65)
