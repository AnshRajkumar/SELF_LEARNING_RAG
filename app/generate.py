# ─────────────────────────────────────────────────────────────────────────────
# app/generate.py  —  Adaptive LLM Generation with Feedback Reflection Memory
#
# Generation Features:
#   1. Grounded Context Generation: Answers queries using retrieved chunks.
#   2. Dynamic In-Context Prompt Adaptation (Step 3 Self-Learning):
#      Automatically retrieves past user feedback, corrections, and failure
#      guidelines from app/exemplars.py matching the query topic, dynamically
#      tuning the LLM system prompt to prevent repeating historical mistakes.
#   3. Dual Providers: Groq LLM (ultra-fast inference) + Mock rule-based generator.
#
# Endpoint: POST /generate
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL
from app.exemplars import retrieve_relevant_exemplars
from app.logger import logger

router = APIRouter()


# ── Dynamic Prompt Tuning (Step 3 Self-Learning) ──────────────────────────

def tune_prompt(query: str, exemplars: Optional[list[dict]] = None) -> tuple[str, bool]:
    """
    Dynamically adapt the system prompt using episodic feedback exemplars.

    When queries on this topic previously failed and user corrections were provided,
    this injects the learned constraints directly into the prompt context.

    Returns:
        (adapted_system_prompt, is_adapted)
    """
    base_prompt = (
        "You are a precise and helpful AI assistant for a RAG (Retrieval-Augmented Generation) system. "
        "Your job is to answer the user's question using ONLY the provided context documents. "
        "Rules:\n"
        "1. Base your answer strictly on the context provided.\n"
        "2. If the context doesn't contain enough information, say so clearly.\n"
        "3. Be concise, accurate, and well-structured.\n"
        "4. Do not hallucinate or add ungrounded claims.\n"
        "5. If multiple context sources are provided, synthesize them into a coherent answer."
    )

    if not exemplars:
        return base_prompt, False

    directives = []
    for i, ex in enumerate(exemplars, start=1):
        directives.append(
            f"[Guideline {i}] For queries similar to '{ex['query']}': "
            f"User correction/directive: \"{ex['correction']}\""
        )

    directives_block = "\n".join(directives)
    adapted_prompt = (
        f"{base_prompt}\n\n"
        "IMPORTANT — Learned Directives from Past User Feedback:\n"
        f"{directives_block}\n"
        "Adhere to these learned directives while answering."
    )

    logger.info(f"Dynamically adapted generation prompt with {len(exemplars)} feedback exemplar(s).")
    return adapted_prompt, True


# ── Mock LLM ──────────────────────────────────────────────────────────────

def mock_generate(query: str, context: str, exemplars: Optional[list[dict]] = None) -> str:
    """
    Rule-based answer generator for development and offline testing.
    Incorporates learned exemplar guidance when context is missing.
    """
    if context.strip():
        answer = (
            f"Based on the retrieved information:\n\n"
            f"{context}\n\n"
            f"To answer your question '{query}':\n"
            f"The above context provides the most relevant information available "
            f"in the knowledge base."
        )
    elif exemplars:
        guidance = " | ".join(ex["correction"] for ex in exemplars)
        answer = (
            f"I couldn't find documents for '{query}' in the current knowledge base. "
            f"However, based on learned user guidance: {guidance}"
        )
    else:
        answer = (
            f"I couldn't find relevant information for '{query}' "
            f"in the knowledge base. Try rephrasing your query or "
            f"adding more documents via POST /ingest."
        )
    return answer


# ── Groq LLM ──────────────────────────────────────────────────────────────

def groq_generate(query: str, context: str, exemplars: Optional[list[dict]] = None) -> str:
    """
    Generate an answer using Groq's LLM inference API with dynamic prompt adaptation.
    """
    try:
        from groq import Groq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env or switch to LLM_PROVIDER=mock."
            )

        client = Groq(api_key=GROQ_API_KEY)

        # Dynamic prompt adaptation using learned reflection memory
        system_prompt, is_adapted = tune_prompt(query, exemplars)

        user_message = (
            f"Context Documents:\n"
            f"{'='*50}\n"
            f"{context if context.strip() else '[No relevant documents found]'}\n"
            f"{'='*50}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        logger.info(f"Calling Groq API | model={GROQ_MODEL} | adapted={is_adapted}")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            stream=False,
        )

        answer = response.choices[0].message.content.strip()

        if hasattr(response, "usage") and response.usage:
            logger.info(
                f"Groq usage — prompt: {response.usage.prompt_tokens} tokens, "
                f"completion: {response.usage.completion_tokens} tokens"
            )

        return answer

    except Exception as e:
        logger.error(f"Groq generation error: {e}. Falling back to mock generator.")
        return mock_generate(query, context, exemplars)


# ── Core Generation Pipeline ───────────────────────────────────────────────

def generate_answer(
    query: str,
    context_docs: list[dict],
    use_exemplars: bool = True,
) -> str:
    """
    Generate an answer from query and context documents.
    Automatically retrieves matching episodic exemplars to adapt the prompt.
    """
    exemplars = []
    if use_exemplars:
        try:
            exemplars = retrieve_relevant_exemplars(query)
        except Exception as e:
            logger.warning(f"Could not retrieve feedback exemplars: {e}")

    # Build numbered context block
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source", "unknown")
        score = doc.get("score", "N/A")
        context_parts.append(
            f"[Document {i} | Source: {source} | Score: {score}]\n{doc['text']}"
        )
    context = "\n\n".join(context_parts)

    logger.info(
        f"Generating answer | provider='{LLM_PROVIDER}' | "
        f"docs={len(context_docs)} | exemplars_applied={len(exemplars)}"
    )

    if LLM_PROVIDER == "groq":
        return groq_generate(query, context, exemplars)
    else:
        return mock_generate(query, context, exemplars)


# ── Pydantic Models ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    context: list[dict] = Field(
        default=[],
        description="Retrieved document dicts (from /retrieve). If empty, generates without context."
    )
    use_exemplars: bool = Field(
        default=True,
        description="Whether to inject learned feedback exemplars into the prompt"
    )


class GenerateResponse(BaseModel):
    query: str
    answer: str
    llm_provider: str
    model: str
    exemplars_used: int = 0


# ── API Endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate an answer using LLM with adaptive feedback prompt tuning"
)
def generate(request: GenerateRequest):
    """
    Generate an answer using retrieved context documents and adaptive prompt tuning:
    - Matches query against learned feedback memory (exemplars.json)
    - Automatically injects past user corrections into the LLM system prompt
    - Prevents repeating known failure patterns
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    exemplars = []
    if request.use_exemplars:
        exemplars = retrieve_relevant_exemplars(request.query)

    answer = generate_answer(
        query=request.query,
        context_docs=request.context,
        use_exemplars=request.use_exemplars,
    )

    model_used = GROQ_MODEL if LLM_PROVIDER == "groq" else "mock-template"

    return GenerateResponse(
        query=request.query,
        answer=answer,
        llm_provider=LLM_PROVIDER,
        model=model_used,
        exemplars_used=len(exemplars),
    )
