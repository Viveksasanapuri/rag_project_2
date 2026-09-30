from pathlib import Path
from typing import Iterable
import time

from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
import requests

from .config import get_settings


def download_pdf() -> Path:
    settings = get_settings()
    path = Path(settings.pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        return path

    print(f"Downloading PDF from {settings.pdf_url}...")
    response = requests.get(settings.pdf_url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Downloaded PDF to {path} ({len(response.content)} bytes).")
    return path


def load_and_chunk_pdf(pdf_path: Path) -> list[dict]:
    settings = get_settings()
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for local_chunk_no, chunk in enumerate(splitter.split_text(text)):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunks.append(
                {
                    "id": f"page-{page_number}-chunk-{local_chunk_no}",
                    "text": chunk,
                    "page": page_number,
                    "chunk_no": local_chunk_no,
                    "source": pdf_path.name,
                }
            )

    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")

    return chunks


def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def get_embeddings(client: InferenceClient, model: str, texts: list[str]) -> list[list[float]]:
    raw = client.feature_extraction(texts, model=model)
    arr = np.array(raw)
    if arr.ndim == 1:
        return [arr.tolist()]
    if arr.ndim == 2:
        return [row.tolist() for row in arr]
    if arr.ndim == 3:
        pooled = arr.mean(axis=1)
        return [row.tolist() for row in pooled]
    raise ValueError(f"Unexpected embedding output shape: {arr.shape}")


def ingest() -> None:
    settings = get_settings()
    pdf_path = download_pdf()
    chunks = load_and_chunk_pdf(pdf_path)
    print(f"Extracted {len(chunks)} text chunks from {pdf_path.name}.")

    hf_client = InferenceClient(api_key=settings.hf_token)
    pinecone = Pinecone(api_key=settings.pinecone_api_key)

    print(f"Generating sample embedding with {settings.hf_embedding_model} to determine dimension...")
    sample_emb = get_embeddings(hf_client, settings.hf_embedding_model, [chunks[0]["text"]])[0]
    dimension = len(sample_emb)
    print(f"Embedding dimension detected: {dimension}")

    if not pinecone.has_index(settings.pinecone_index_name):
        print(f"Creating Pinecone index '{settings.pinecone_index_name}' (dimension={dimension}, metric=cosine)...")
        pinecone.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Pinecone index created.")
    else:
        index_desc = pinecone.describe_index(settings.pinecone_index_name)
        if index_desc.dimension != dimension:
            print(
                f"Existing index has dimension {index_desc.dimension}, but model uses {dimension}. "
                f"Recreating index '{settings.pinecone_index_name}'..."
            )
            pinecone.delete_index(settings.pinecone_index_name)
            time.sleep(2)
            pinecone.create_index(
                name=settings.pinecone_index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("Pinecone index recreated.")

    index = pinecone.Index(settings.pinecone_index_name)

    print(f"Clearing namespace '{settings.pinecone_namespace}' before ingestion...")
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    except Exception:
        pass

    batch_size = 16
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Upserting {len(chunks)} chunks into Pinecone in {total_batches} batches of {batch_size}...")

    for batch_idx, batch in enumerate(batched(chunks, batch_size), start=1):
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(hf_client, settings.hf_embedding_model, texts)

        vectors = []
        for item, emb in zip(batch, embeddings):
            vectors.append(
                {
                    "id": item["id"],
                    "values": emb,
                    "metadata": {
                        "text": item["text"],
                        "page": item["page"],
                        "chunk_no": item["chunk_no"],
                        "source": item["source"],
                    },
                }
            )

        index.upsert(
            vectors=vectors,
            namespace=settings.pinecone_namespace,
        )
        print(f"  Batch {batch_idx}/{total_batches} upserted ({len(vectors)} vectors).")

    print(f"Successfully ingested {len(chunks)} chunks into Pinecone.")
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Namespace: {settings.pinecone_namespace}")
    print(f"Embedding Model: {settings.hf_embedding_model}")


if __name__ == "__main__":
    ingest()
