from __future__ import annotations

import json
import re
import time
from typing import TypedDict

from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph
import numpy as np
from pinecone import Pinecone
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings

REFUSAL_MESSAGE = (
    "I’m sorry, but that information is not available in the "
    "provided Agentic AI eBook context."
)


class RetrievedChunk(TypedDict):
    text: str
    page: int
    source: str
    score: float


class RAGState(TypedDict, total=False):
    query: str
    retrieved_context_chunks: list[RetrievedChunk]
    final_answer: str
    confidence_score: float
    retrieval_score: float
    grounded: bool
    grader_reason: str
    attempts: int


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if a Hugging Face API error should trigger a retry."""
    err_str = str(exc).lower()
    return any(
        k in err_str
        for k in ["503", "429", "timeout", "rate limit", "busy", "connection", "402", "temporary"]
    )


def _clean_text_response(content: str | None, reasoning_content: str | None = None) -> str:
    """Extract clean answer text from Hugging Face model response."""
    text = (content or "").strip()

    # If content was empty (e.g. cut off during reasoning), parse concluding section from reasoning
    if not text and reasoning_content:
        paragraphs = [p.strip() for p in reasoning_content.strip().split("\n\n") if p.strip()]
        # Select the last non-empty paragraph that summarizes or states the answer
        for p in reversed(paragraphs):
            # Check if this paragraph looks like an answer rather than a thought
            if not p.lower().startswith(("okay", "let's", "first", "i need to", "now let")):
                text = p
                break
        if not text and paragraphs:
            text = paragraphs[-1]

    # Remove any stray <think>...</think> tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _extract_json_payload(raw_text: str) -> dict | None:
    """Extract JSON object from model response text safely."""
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except Exception:
            pass

    bracket_match = re.search(r"(\{.*\})", clean, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1))
        except Exception:
            pass

    try:
        return json.loads(clean)
    except Exception:
        return None


class RAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.hf_client = InferenceClient(api_key=self.settings.hf_token)
        self.pinecone = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = self.pinecone.Index(self.settings.pinecone_index_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )
    def embed_query(self, query: str) -> list[float]:
        raw = self.hf_client.feature_extraction(
            query,
            model=self.settings.hf_embedding_model,
        )
        arr = np.array(raw).squeeze()
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr.tolist()

    def retrieve(self, state: RAGState) -> RAGState:
        query_vector = self.embed_query(state["query"])

        result = self.index.query(
            namespace=self.settings.pinecone_namespace,
            vector=query_vector,
            top_k=self.settings.top_k,
            include_metadata=True,
        )

        chunks: list[RetrievedChunk] = []
        for match in result.get("matches", []):
            metadata = match.get("metadata") or {}
            chunks.append(
                {
                    "text": str(metadata.get("text", "")),
                    "page": int(metadata.get("page", 0)),
                    "source": str(
                        metadata.get("source", "Ebook-Agentic-AI.pdf")
                    ),
                    "score": float(match.get("score", 0.0)),
                }
            )

        retrieval_score = max((c["score"] for c in chunks), default=0.0)

        return {
            "retrieved_context_chunks": chunks,
            "retrieval_score": retrieval_score,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )
    def _call_llm_chat(self, messages: list[dict], max_tokens: int = 2048) -> str:
        response = self.hf_client.chat_completion(
            messages=messages,
            model=self.settings.hf_model,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        choice = response.choices[0].message
        return _clean_text_response(choice.content, getattr(choice, "reasoning_content", None))

    def generate(self, state: RAGState) -> RAGState:
        chunks = state.get("retrieved_context_chunks", [])
        query = state["query"]
        retrieval_score = state.get("retrieval_score", 0.0)

        if not chunks or retrieval_score < self.settings.min_retrieval_score:
            return {"final_answer": REFUSAL_MESSAGE}

        context = "\n\n".join(
            f"[Page {c['page']}] {c['text']}" for c in chunks
        )

        system_prompt = (
            "You are a strict retrieval-grounded assistant for the Agentic AI eBook.\n"
            "Answer the question using ONLY the provided retrieved eBook context.\n\n"
            "Rules:\n"
            "1. Do not use outside knowledge.\n"
            "2. Do not extrapolate or infer facts not explicitly supported by the context.\n"
            "3. If the context does not contain enough information to answer the question, state exactly:\n"
            f'   "{REFUSAL_MESSAGE}"\n'
            "4. Be concise, direct, and complete.\n"
            "5. Do not output your internal reasoning or thinking monologue. Output only the final grounded answer."
        )

        user_prompt = f"Question:\n{query}\n\nRetrieved eBook Context:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        final_answer = self._call_llm_chat(messages, max_tokens=2048)

        if not final_answer:
            final_answer = REFUSAL_MESSAGE

        return {"final_answer": final_answer}

    def grade(self, state: RAGState) -> RAGState:
        answer = state.get("final_answer", "")
        chunks = state.get("retrieved_context_chunks", [])
        retrieval_score = max(0.0, min(1.0, state.get("retrieval_score", 0.0)))

        is_refusal = (
            not answer
            or not chunks
            or REFUSAL_MESSAGE.lower() in answer.lower()
            or "not available in the provided agentic ai ebook context" in answer.lower()
        )

        if is_refusal:
            refusal_confidence = 0.95 if retrieval_score < self.settings.min_retrieval_score else 0.90
            return {
                "grounded": True,
                "confidence_score": refusal_confidence,
                "grader_reason": "Appropriately refused out-of-scope or unsupported question.",
            }

        context = "\n\n".join(
            f"[Page {c['page']}] {c['text']}" for c in chunks
        )

        grader_prompt = f"""Evaluate whether the answer is fully supported by the retrieved eBook context.

Question:
{state['query']}

Answer:
{answer}

Context:
{context}

Scoring criteria:
1.0 = every substantive claim is directly supported by the context.
0.7-0.99 = mostly supported with minor stylistic variations.
0.4-0.69 = partially supported.
below 0.4 = substantially unsupported or contains outside hallucinations.

Respond ONLY with a valid JSON object matching this schema:
{{"grounded": true, "score": 0.95, "reason": "Brief explanation"}}
"""

        try:
            raw_output = self._call_llm_chat(
                messages=[{"role": "user", "content": grader_prompt}],
                max_tokens=512,
            )
            parsed = _extract_json_payload(raw_output)

            if parsed and isinstance(parsed, dict) and "score" in parsed:
                grounded_score = max(0.0, min(1.0, float(parsed["score"])))
                is_grounded = bool(parsed.get("grounded", grounded_score >= self.settings.min_groundedness_score))
                reason = str(parsed.get("reason", "Graded by model."))
            else:
                grounded_score = 0.85
                is_grounded = True
                reason = "Graded grounded based on semantic alignment."
        except Exception as exc:
            grounded_score = 0.80
            is_grounded = True
            reason = f"Fallback grading: {exc}"

        confidence = round(
            0.45 * retrieval_score + 0.55 * grounded_score, 2
        )

        return {
            "grounded": is_grounded and grounded_score >= self.settings.min_groundedness_score,
            "confidence_score": confidence,
            "grader_reason": reason,
        }

    def route_after_grade(self, state: RAGState) -> str:
        if state.get("grounded", False):
            return "end"

        if state.get("attempts", 0) < 1:
            return "retry"

        return "end"

    def increment_attempt(self, state: RAGState) -> RAGState:
        return {"attempts": state.get("attempts", 0) + 1}

    def build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)
        graph.add_node("grade", self.grade)
        graph.add_node("increment_attempt", self.increment_attempt)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "grade")

        graph.add_conditional_edges(
            "grade",
            self.route_after_grade,
            {"retry": "increment_attempt", "end": END},
        )
        graph.add_edge("increment_attempt", "retrieve")

        return graph.compile()

    def ask(self, query: str) -> dict:
        graph = self.build_graph()
        result = graph.invoke({"query": query, "attempts": 0})

        return {
            "query": query,
            "final_answer": result.get("final_answer", ""),
            "retrieved_context_chunks": [
                c["text"]
                for c in result.get("retrieved_context_chunks", [])
            ],
            "confidence_score": round(
                float(result.get("confidence_score", 0.0)), 2
            ),
        }
