from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import RAGPipeline


app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    version="1.0.0",
    description="LangGraph + Pinecone + Hugging Face (Qwen3) grounded RAG chatbot.",
)

_pipeline: RAGPipeline | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question about the Agentic AI eBook")


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {
        "title": "Agentic AI eBook RAG Chatbot API",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "query": "POST /query",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    global _pipeline

    try:
        if _pipeline is None:
            _pipeline = RAGPipeline()
        return _pipeline.ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
