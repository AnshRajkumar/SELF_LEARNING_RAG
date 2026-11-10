import json

BASE = "http://127.0.0.1:8000"
_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            import urllib.request
            req = urllib.request.Request(f"{BASE}/", method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as r:
                pass
            _client = "live"
        except Exception:
            from fastapi.testclient import TestClient
            from app.db import init_db
            from app.main import app
            init_db()
            _client = TestClient(app)
    return _client

def post(path, data):
    cl = _get_client()
    if cl == "live":
        import urllib.request
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{BASE}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    else:
        resp = cl.post(path, json=data)
        return resp.json()

def get(path):
    cl = _get_client()
    if cl == "live":
        import urllib.request
        with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
            return json.loads(r.read())
    else:
        resp = cl.get(path)
        return resp.json()

print("=" * 60)
print("  SELF-LEARNING RAG — FULL END-TO-END TEST SUITE")
print("=" * 60)

# TEST 1: Health check
print("\n[TEST 1] GET / — Health Check")
h = get("/")
print(f"  status       : {h['status']}")
print(f"  llm_provider : {h['llm_provider']}")
print(f"  embed_model  : {h['embedding_model']}")
assert h["status"] == "ok"
print("  PASS")

# TEST 2: Retrieve
print("\n[TEST 2] POST /retrieve — Semantic Search")
r = post("/retrieve", {"query": "What is machine learning?", "top_k": 2})
print(f"  docs found : {r['total_found']}")
for d in r["results"]:
    print(f"  [{d['rank']}] score={d['score']} | {d['text'][:70]}...")
assert r["total_found"] > 0
print("  PASS")

# TEST 3: Full RAG pipeline with Groq
print("\n[TEST 3] POST /ask — Full RAG Pipeline (Groq LLM)")
a = post("/ask", {"query": "What is retrieval-augmented generation?", "top_k": 3})
print(f"  query_id     : {a['query_id']}")
print(f"  docs_used    : {a['pipeline']['docs_retrieved']}")
print(f"  retrieve_ms  : {a['pipeline']['retrieve_ms']}ms")
print(f"  generate_ms  : {a['pipeline']['generate_ms']}ms")
print(f"  total_ms     : {a['pipeline']['total_ms']}ms")
print(f"  quality      : {a['auto_eval']['label']} (score={a['auto_eval']['score']})")
print(f"  answer       : {a['answer'][:250]}...")
assert a["answer"]
print("  PASS")

# TEST 4: Evaluate (Standalone)
print("\n[TEST 4] POST /evaluate — Standalone Response Relevancy")
e = post("/evaluate", {
    "query": "What is retrieval-augmented generation?",
    "answer": a["answer"]
})
print(f"  score   : {e['score']}")
print(f"  label   : {e['label']}")
print(f"  details : {e['details']}")
assert "score" in e
print("  PASS")

# TEST 4b: Full RAG Triad Evaluation (With Retrieved Context)
print("\n[TEST 4b] POST /evaluate — Full RAG Triad (Faithfulness, Relevancy, Precision)")
e_triad = post("/evaluate", {
    "query": a["query"],
    "answer": a["answer"],
    "context": a["sources"]
})
print(f"  engine            : {e_triad.get('engine')}")
print(f"  composite score   : {e_triad['score']}")
print(f"  faithfulness      : {e_triad['metrics']['faithfulness']}")
print(f"  answer_relevancy  : {e_triad['metrics']['answer_relevancy']}")
print(f"  context_precision : {e_triad['metrics']['context_precision']}")
print(f"  details           : {e_triad['details']}")
assert e_triad["metrics"]["faithfulness"] is not None
assert e_triad["metrics"]["context_precision"] is not None
print("  PASS")

# TEST 5: Good feedback
print("\n[TEST 5] POST /feedback — Submit GOOD feedback")
f1 = post("/feedback", {
    "query": a["query"],
    "context": a["sources"],
    "answer": a["answer"],
    "feedback": "good",
    "auto_score": a["auto_eval"]["score"]
})
print(f"  id      : {f1['id']}")
print(f"  message : {f1['message']}")
assert f1["status"] == "success"
print("  PASS")

# TEST 6: Bad feedback (self-learning triggered)
print("\n[TEST 6] POST /feedback — Submit BAD feedback (self-learning)")
f2 = post("/feedback", {
    "query": "What is quantum computing?",
    "context": [],
    "answer": "I could not find relevant information.",
    "feedback": "bad",
    "auto_score": 0.1
})
print(f"  id      : {f2['id']}")
print(f"  message : {f2['message']}")
assert f2["status"] == "success"
print("  PASS")

# TEST 7: Feedback stats
print("\n[TEST 7] GET /feedback/stats — Aggregate Statistics")
s = get("/feedback/stats")
print(f"  total    : {s['total']}")
print(f"  good     : {s['good']} ({s['good_pct']}%)")
print(f"  bad      : {s['bad']} ({s['bad_pct']}%)")
print(f"  avg_score: {s['avg_score']}")
assert s["total"] >= 2
print("  PASS")

# TEST 8: Improvement hints
print("\n[TEST 8] GET /feedback/improvements — Self-Learning Hints")
imp = get("/feedback/improvements")
print(f"  total_flagged      : {imp['total_flagged']}")
print(f"  needs_retraining   : {imp['retraining_signal']['needs_retraining']}")
print(f"  bad_feedback_pct   : {imp['retraining_signal']['bad_feedback_pct']}%")
if imp["flagged_queries"]:
    for fq in imp["flagged_queries"]:
        print(f"  flagged: '{fq['query']}' — {fq['recommended_action']}")
print("  PASS")

print("\n" + "=" * 60)
print("  ALL 8 TESTS PASSED — SYSTEM IS FULLY OPERATIONAL")
print("=" * 60)
