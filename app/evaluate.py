# ─────────────────────────────────────────────────────────────────────────────
# app/evaluate.py  —  Research-Grade RAG Triad Evaluation (Ragas-style metrics)
#
# Replaces naive keyword-overlap with multi-dimensional evaluation:
#   1. Faithfulness (Groundedness): Are claims in the answer strictly supported
#      by the retrieved context? (Hallucination detection)
#   2. Answer Relevancy: Does the answer directly address the user's question
#      without off-topic drift or evasiveness?
#   3. Context Precision: Are the retrieved passages relevant to the question,
#      and are the most relevant chunks ranked at the top?
#
# Supports dual evaluation engines:
#   - LLM Judge (Primary): Uses Groq LLM (e.g., Qwen/LLaMA) with structured
#     claim decomposition and verification prompts (standard Ragas methodology).
#   - Semantic Fallback: Uses Sentence-Transformers embeddings (all-MiniLM-L6-v2)
#     to compute cosine similarity, rank precision, and sentence-level entailment
#     deterministically with zero API cost.
#
# Endpoint: POST /evaluate
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GOOD_SCORE_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    WEIGHT_FAITHFULNESS,
    WEIGHT_ANSWER_RELEVANCY,
    WEIGHT_CONTEXT_PRECISION,
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from app.logger import logger

router = APIRouter()


# ── Context Normalization ──────────────────────────────────────────────────

def _normalize_context(context: Optional[Union[list[dict], list[str], str]]) -> list[str]:
    """
    Normalize various context formats (list of dicts from /ask, list of strings,
    or a single string) into a clean list of text chunks.
    """
    if not context:
        return []
    if isinstance(context, str):
        # Split single block into non-empty paragraphs
        chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
        return chunks if chunks else [context.strip()]
    if isinstance(context, list):
        normalized = []
        for item in context:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text.strip():
                    normalized.append(text.strip())
            elif isinstance(item, str) and item.strip():
                normalized.append(item.strip())
        return normalized
    return []


# ── Semantic Embedding Fallback Engine ─────────────────────────────────────

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for atomic claim analysis."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def _evaluate_semantic_fallback(
    query: str,
    answer: str,
    context_chunks: list[str],
) -> dict:
    """
    Deterministic zero-cost evaluation using sentence-transformers embeddings.
    Calculates Ragas-aligned proxy metrics via vector similarities.
    """
    from app.ingest import get_model

    model = get_model()
    query_emb = model.encode(query, convert_to_numpy=True)
    answer_emb = model.encode(answer, convert_to_numpy=True)

    # 1. Answer Relevancy: semantic similarity between query and answer
    raw_sim = _cosine_similarity(query_emb, answer_emb)
    # Scale from typical range [0.1, 0.85] to [0.0, 1.0]
    scaled_relevancy = max(0.0, min(1.0, (raw_sim - 0.15) / 0.70))
    answer_relevancy = round(scaled_relevancy, 4)

    # 2. Context Precision & Faithfulness (requires context chunks)
    if not context_chunks:
        return {
            "faithfulness": None,
            "answer_relevancy": answer_relevancy,
            "context_precision": None,
            "engine": "semantic_fallback",
            "critique": "Context was not provided; evaluated answer relevancy only.",
            "claims": [],
        }

    chunk_embs = model.encode(context_chunks, convert_to_numpy=True)

    # Context Precision: precision@k weighted by rank
    # A chunk is relevant if cosine_sim(query, chunk) >= 0.35
    relevant_indicators = []
    chunk_sims = []
    for i, c_emb in enumerate(chunk_embs):
        sim = _cosine_similarity(query_emb, c_emb)
        chunk_sims.append(sim)
        relevant_indicators.append(1 if sim >= 0.35 else 0)

    total_relevant = sum(relevant_indicators)
    if total_relevant == 0:
        context_precision = 0.0
    else:
        # Standard Ragas Context Precision formula: sum(Precision@k * rel_k) / total_relevant
        precision_sum = 0.0
        running_rel = 0
        for k, rel in enumerate(relevant_indicators, start=1):
            if rel == 1:
                running_rel += 1
                precision_sum += (running_rel / k)
        context_precision = round(min(1.0, precision_sum / total_relevant), 4)

    # Faithfulness: sentence entailment against context chunks
    sentences = _split_into_sentences(answer)
    if not sentences:
        faithfulness = 1.0
        claims = []
    else:
        sent_embs = model.encode(sentences, convert_to_numpy=True)
        supported_count = 0
        claims = []
        for s_idx, s_text in enumerate(sentences):
            # Check maximum similarity of this sentence across all context chunks
            max_chunk_sim = max(
                _cosine_similarity(sent_embs[s_idx], c_emb) for c_emb in chunk_embs
            )
            is_supported = max_chunk_sim >= 0.50
            if is_supported:
                supported_count += 1
            claims.append({
                "statement": s_text,
                "status": "SUPPORTED" if is_supported else "UNGROUNDED",
                "similarity": round(max_chunk_sim, 3),
            })
        faithfulness = round(supported_count / len(sentences), 4)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "engine": "semantic_fallback",
        "critique": (
            f"Embedding-based evaluation: {total_relevant}/{len(context_chunks)} chunks relevant, "
            f"{supported_count if sentences else 1}/{len(sentences) if sentences else 1} statements grounded."
        ),
        "claims": claims,
    }


# ── LLM Judge Engine (Groq / Ragas Method) ─────────────────────────────────

def _evaluate_llm_judge(
    query: str,
    answer: str,
    context_chunks: list[str],
) -> Optional[dict]:
    """
    Ragas-style LLM-as-a-judge evaluation using Groq.
    Performs atomic claim decomposition, factual verification, and context ranking.
    """
    if LLM_PROVIDER != "groq" or not GROQ_API_KEY:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        formatted_context = ""
        if context_chunks:
            formatted_context = "\n\n".join(
                f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
            )
        else:
            formatted_context = "[No retrieved context provided]"

        system_prompt = (
            "You are an expert AI evaluator for RAG (Retrieval-Augmented Generation) systems. "
            "You rigorously score outputs according to the RAG Triad framework:\n"
            "1. Faithfulness (0.0 to 1.0): Are all factual statements in the answer strictly supported by the context? "
            "   (Flag hallucinations or external knowledge not found in the context). "
            "   If the answer accurately admits lack of information based on context, score 1.0.\n"
            "2. Answer Relevancy (0.0 to 1.0): Does the answer directly and concisely address the user's question without drifting?\n"
            "3. Context Precision (0.0 to 1.0): Did the retrieved chunks contain information relevant to answering the question?\n\n"
            "Return ONLY a valid JSON object with this exact schema:\n"
            "{\n"
            '  "faithfulness": {"score": float, "claims": [{"statement": str, "status": "SUPPORTED" | "UNGROUNDED", "reason": str}], "reason": str},\n'
            '  "answer_relevancy": {"score": float, "reason": str},\n'
            '  "context_precision": {"score": float, "relevant_chunks": [int], "reason": str},\n'
            '  "critique": str\n'
            "}"
        )

        user_content = (
            f"Question:\n{query}\n\n"
            f"Context Documents:\n{formatted_context}\n\n"
            f"Generated Answer to Evaluate:\n{answer}"
        )

        logger.info(f"Calling LLM Judge ({GROQ_MODEL}) for RAG Triad evaluation")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,  # Low temperature for deterministic scoring
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        f_data = data.get("faithfulness", {})
        r_data = data.get("answer_relevancy", {})
        p_data = data.get("context_precision", {})

        f_score = float(f_data.get("score", 0.8)) if context_chunks else None
        r_score = float(r_data.get("score", 0.8))
        p_score = float(p_data.get("score", 0.8)) if context_chunks else None

        return {
            "faithfulness": round(max(0.0, min(1.0, f_score)), 4) if f_score is not None else None,
            "answer_relevancy": round(max(0.0, min(1.0, r_score)), 4),
            "context_precision": round(max(0.0, min(1.0, p_score)), 4) if p_score is not None else None,
            "engine": "llm_judge",
            "critique": data.get("critique") or r_data.get("reason", "Evaluation complete."),
            "claims": f_data.get("claims", []),
        }

    except Exception as e:
        logger.warning(f"LLM judge evaluation failed ({e}). Falling back to semantic engine.")
        return None


# ── Core Evaluation Orchestrator ───────────────────────────────────────────

def evaluate_response(
    query: str,
    answer: str,
    context: Optional[Union[list[dict], list[str], str]] = None,
) -> dict:
    """
    Evaluate response quality across the RAG Triad:
      - Faithfulness (groundedness in context)
      - Answer Relevancy (directness to question)
      - Context Precision (retrieval chunk quality)

    Args:
        query:   The original user question.
        answer:  The generated response from the LLM.
        context: Optional retrieved context chunks (dicts, strings, or text block).

    Returns:
        Dict with composite score, quality label ("good"/"bad"), metrics, and diagnostic details.
    """
    context_chunks = _normalize_context(context)

    # 1. Attempt LLM judge first; fall back to deterministic semantic engine
    eval_result = _evaluate_llm_judge(query, answer, context_chunks)
    if not eval_result:
        eval_result = _evaluate_semantic_fallback(query, answer, context_chunks)

    faithfulness = eval_result["faithfulness"]
    answer_relevancy = eval_result["answer_relevancy"]
    context_precision = eval_result["context_precision"]

    # 2. Compute composite RAG Triad score
    if faithfulness is not None and context_precision is not None:
        composite_score = (
            (WEIGHT_FAITHFULNESS * faithfulness)
            + (WEIGHT_ANSWER_RELEVANCY * answer_relevancy)
            + (WEIGHT_CONTEXT_PRECISION * context_precision)
        )
    else:
        # Fallback when context was not provided (evaluate answer relevancy only)
        composite_score = answer_relevancy

    composite_score = round(max(0.0, min(1.0, composite_score)), 4)

    # 3. Determine quality verdict ("good" vs "bad")
    # Rule: Even if relevancy is high, a severe hallucination (low faithfulness) yields "bad"
    is_hallucination = (
        faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD
    )
    if is_hallucination:
        label = "bad"
        verdict_reason = f"Flagged as ungrounded/hallucinated (Faithfulness: {faithfulness:.1%} < {FAITHFULNESS_THRESHOLD:.0%})"
    elif composite_score >= GOOD_SCORE_THRESHOLD:
        label = "good"
        verdict_reason = f"Passed all RAG Triad quality thresholds ({composite_score:.1%} >= {GOOD_SCORE_THRESHOLD:.0%})"
    else:
        label = "bad"
        verdict_reason = f"Score below quality threshold ({composite_score:.1%} < {GOOD_SCORE_THRESHOLD:.0%})"

    # 4. Format descriptive summary string
    if faithfulness is not None and context_precision is not None:
        details = (
            f"RAG Triad [{eval_result['engine']}]: "
            f"Faithfulness={faithfulness:.1%}, "
            f"Relevancy={answer_relevancy:.1%}, "
            f"Precision={context_precision:.1%}. "
            f"Composite={composite_score:.1%} -> {label.upper()} ({verdict_reason})."
        )
    else:
        details = (
            f"RAG Evaluation [{eval_result['engine']}]: "
            f"Relevancy={answer_relevancy:.1%}. "
            f"(Context omitted). Composite={composite_score:.1%} -> {label.upper()}."
        )

    logger.info(
        f"RAG Evaluation ({eval_result['engine']}) — "
        f"composite={composite_score:.3f}, label={label}, "
        f"faithfulness={faithfulness}, relevancy={answer_relevancy}, precision={context_precision}"
    )

    return {
        "score": composite_score,
        "label": label,
        "details": details,
        "threshold": GOOD_SCORE_THRESHOLD,
        "metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        },
        "engine": eval_result["engine"],
        "critique": eval_result.get("critique", ""),
        "claims": eval_result.get("claims", []),
    }


# ── Pydantic Request & Response Models ─────────────────────────────────────

class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Original user query")
    answer: str = Field(..., min_length=1, description="Generated answer to evaluate")
    context: Optional[Union[list[dict], list[str], str]] = Field(
        default=None,
        description="Retrieved context chunks (optional, enables Faithfulness & Context Precision checks)"
    )


class EvaluateResponse(BaseModel):
    query: str
    answer: str
    score: float = Field(..., description="Overall composite RAG Triad score (0.0 to 1.0)")
    label: str = Field(..., description="'good' or 'bad'")
    details: str = Field(..., description="Diagnostic summary string")
    threshold: float
    metrics: dict[str, Any] = Field(
        ...,
        description="Detailed RAG Triad scores: faithfulness, answer_relevancy, context_precision"
    )
    engine: str = Field(..., description="Engine used: 'llm_judge' or 'semantic_fallback'")
    critique: Optional[str] = None
    claims: Optional[list[dict]] = None


# ── API Endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate quality of a generated answer using the RAG Triad (Ragas-aligned)"
)
def evaluate(request: EvaluateRequest):
    """
    Evaluate a generated answer using the **RAG Triad** framework:
    - **Faithfulness (Groundedness)**: Checks whether facts in the answer are supported by retrieved context.
    - **Answer Relevancy**: Checks whether the answer directly addresses the query.
    - **Context Precision**: Checks whether the retrieved passages were relevant and well-ranked.

    Uses Groq LLM as an evaluation judge with automatic fallback to sentence-transformers semantic embeddings.
    """
    result = evaluate_response(
        query=request.query,
        answer=request.answer,
        context=request.context,
    )
    return EvaluateResponse(
        query=request.query,
        answer=request.answer,
        **result,
    )
