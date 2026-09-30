# Agentic AI eBook — LangGraph & Pinecone RAG Chatbot

A high-performance, strictly grounded Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Pinecone**, and **Hugging Face** (`Qwen/Qwen3-8B` + `Qwen/Qwen3-Embedding-8B`). Answers questions strictly grounded in the official [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).

---

## 1. System Architecture

```text
               +----------------------------------+
               |  Agentic AI eBook PDF (60 pages) |
               +-----------------+----------------+
                                 |
                                 v
                     [ PyPDF Text Extraction ]
                                 |
                                 v
               [ Recursive Character Text Splitter ]
                     (chunk: 800, overlap: 120)
                                 |
                                 v
                [ Hugging Face Inference API ]
                   (Qwen/Qwen3-Embedding-8B)
                        (4096 dimensions)
                                 |
                                 v
                    +------------------------+
                    |    Pinecone Vector DB  |
                    | (Serverless: us-east-1)|
                    +-----------+------------+
                                |
                     Vector Similarity Search
                                |
                                v
               +----------------------------------+
               |     LangGraph Orchestration      |
               |                                  |
               |   [START] ➔ [retrieve]           |
               |                |                 |
               |                v                 |
               |           [generate]             |
               |          (Qwen3-8B)              |
               |                |                 |
               |                v                 |
               |             [grade]              |
               |        (Grounding Check)         |
               |             /     \              |
               |   (grounded)       (ungrounded)  |
               |       v                  v       |
               |     [END]      [increment_attempt]
               |                          |       |
               |                          v       |
               |                      [retrieve]  |
               +----------------+-----------------+
                                |
                                v
                +---------------+---------------+
                |                               |
                v                               v
        [ FastAPI REST API ]          [ Streamlit Web UI ]
         POST /query (:8000)             Interactive (:8501)
```

### Key Workflow Highlights
1. **Document Ingestion**: Automatically downloads `Ebook-Agentic-AI.pdf` (60 pages), cleans and segments text into 136 overlapping chunks with page metadata.
2. **Dense Vector Embeddings**: Uses `Qwen/Qwen3-Embedding-8B` to produce 4096-dimensional embeddings indexed in Pinecone Serverless.
3. **Graph-based Orchestration (LangGraph)**:
   - **`retrieve`**: Searches Pinecone for top-$k$ relevant chunks using cosine similarity.
   - **`generate`**: Synthesizes an answer using `Qwen/Qwen3-8B` conditioned **strictly** on retrieved context.
   - **`grade`**: Evaluates hallucination and groundedness; checks whether claims are fully supported.
   - **Cyclic Fallback / Bounded Retry**: If ungrounded, routes through a single bounded retry before concluding.
4. **Strict Refusal of Out-of-Scope Queries**: If question is not answerable from the eBook, returns a clean refusal message with high confidence.

---

## 2. API & Response Specification

Every query via FastAPI or Streamlit produces the exact structured JSON response required by the specification:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to autonomous systems capable of perception, reasoning, and goal execution...",
  "retrieved_context_chunks": [
    "At its core, Agentic AI is about autonomous systems that perceive...",
    "Agentic AI provides a practical framework for leveraging these systems..."
  ],
  "confidence_score": 0.89
}
```

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.11+
- Hugging Face API Token (with access to inference models)
- Pinecone API Key

### Setup Environment

```bash
# Clone the repository
git clone <YOUR_PUBLIC_GITHUB_REPO_URL>
cd agentic_ai_rag_chatbot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables
Copy `.env.example` to `.env` and enter your credentials:

```bash
cp .env.example .env
```

Your `.env` should look like:
```ini
HF_TOKEN=hf_your_hugging_face_token
HF_MODEL=Qwen/Qwen3-8B
HF_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
PINECONE_NAMESPACE=agentic-ai-book

TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
MIN_RETRIEVAL_SCORE=0.35
MIN_GROUNDEDNESS_SCORE=0.70

PDF_URL=https://konverge.ai/pdf/Ebook-Agentic-AI.pdf
PDF_PATH=data/Ebook-Agentic-AI.pdf
```

---

## 4. Running the Project

### Step 1: Ingest the eBook into Pinecone

```bash
python -m scripts.ingest
```

This will:
1. Download `Ebook-Agentic-AI.pdf` into `data/`.
2. Extract text across all 60 pages and split into chunks.
3. Compute 4096-dimensional embeddings with `Qwen/Qwen3-Embedding-8B`.
4. Create the Pinecone index if not present and upsert all vectors with source and page metadata.

### Step 2: Run the FastAPI REST Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API Root: `http://127.0.0.1:8000/`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Step 3: Run the Streamlit Web Application

```bash
streamlit run app/ui.py
```

Opens a visual dashboard with:
- Chat interface with chat history
- Dynamic confidence score display
- Retrieved context chunk inspector with page numbers
- Raw JSON response payload viewer
- Preset buttons for the 6 assignment validation queries

### Step 4: Run the Automated Validation Suite

```bash
python -m scripts.validate_queries
```

Runs all 6 core assignment queries (both in-scope and out-of-scope) and prints the results.

### Step 5: Run Unit Tests

```bash
python -m pytest tests/
```

---

## 5. Sample Query Testing (cURL)

### In-Scope Query:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the core definition of Agentic AI as outlined in the eBook?"}'
```

### Out-of-Scope Query (Groundedness Check):
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

**Expected Response**:
```json
{
  "query": "What is the capital of France?",
  "final_answer": "I’m sorry, but that information is not available in the provided Agentic AI eBook context.",
  "retrieved_context_chunks": [...],
  "confidence_score": 0.95
}
```

---

## 6. Project Structure

```text
agentic_ai_rag_chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py           # Pydantic settings loading from .env
│   ├── ingest.py           # PDF ingestion, chunking, and HF embeddings to Pinecone
│   ├── rag.py              # LangGraph RAG pipeline (Retrieve, Generate, Grade)
│   ├── main.py             # FastAPI REST endpoints
│   └── ui.py               # Streamlit interactive Web UI
├── data/
│   └── Ebook-Agentic-AI.pdf# Downloaded eBook
├── scripts/
│   ├── ingest.py           # Ingestion CLI trigger
│   └── validate_queries.py # 6-query automated test validation script
├── tests/
│   └── test_api.py         # Pytest API test suite
├── .env.example            # Environment variables template
├── requirements.txt        # Python package dependencies
├── RUN_INSTRUCTIONS.md     # Detailed step-by-step commands and usage guide
└── README.md               # Architecture, setup, and overview
```

---

## 7. Technical Details & Design Decisions

- **Why Qwen/Qwen3-8B & Qwen/Qwen3-Embedding-8B?**
  - High reasoning capability and superior retrieval representation without vendor lock-in to OpenAI.
  - Native 4096-dimension embeddings capturing deep contextual semantics.
- **Why LangGraph StateGraph?**
  - Explicit control over transitions: retrieval, LLM synthesis, hallucination grading, and bounded retries are observable Python nodes rather than opaque black-box chains.
- **Hallucination Prevention**:
  1. Context strictness prompt instruction forbidding outside assumptions.
  2. Independent grounding grader verifying that every claim matches retrieved chunks.
  3. Minimum retrieval similarity score threshold: out-of-scope queries fail retrieval checks and trigger deterministic refusal immediately.
