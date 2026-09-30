from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str = Field(..., alias="HF_TOKEN")
    hf_model: str = Field("Qwen/Qwen3-8B", alias="HF_MODEL")
    hf_embedding_model: str = Field(
        "Qwen/Qwen3-Embedding-8B", alias="HF_EMBEDDING_MODEL"
    )

    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("agentic-ai-rag", alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field("agentic-ai-book", alias="PINECONE_NAMESPACE")

    top_k: int = Field(5, alias="TOP_K")
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")
    min_retrieval_score: float = Field(0.35, alias="MIN_RETRIEVAL_SCORE")
    min_groundedness_score: float = Field(0.70, alias="MIN_GROUNDEDNESS_SCORE")

    pdf_url: str = Field(
        "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf", alias="PDF_URL"
    )
    pdf_path: str = Field("data/Ebook-Agentic-AI.pdf", alias="PDF_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
