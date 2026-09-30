# Agentic AI eBook RAG Chatbot — Complete Execution & Run Instructions

This guide provides step-by-step instructions and commands to set up, ingest, run, test, and validate the **LangGraph + Pinecone + Hugging Face RAG Chatbot**.

---

## 1. Prerequisites

Before running the project, verify that you have:
- **Python 3.11+** installed (tested on Python 3.11, 3.12, 3.14)
- **Hugging Face Account & API Token** (configured in `.env`)
- **Pinecone Account & API Key** (configured in `.env`)
- **Internet connectivity** to connect to Hugging Face Inference API and Pinecone Serverless

---

## 2. Environment Configuration (`.env`)

The project uses `.env` for all sensitive credentials and configuration hyperparameters.

Verify your `.env` file exists in the project root:
```ini
# Hugging Face Settings
HF_TOKEN=your_hugging_face_token
HF_MODEL=Qwen/Qwen3-8B
HF_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

# Pinecone Settings
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
PINECONE_NAMESPACE=agentic-ai-book

# RAG & Chunking Parameters
TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
MIN_RETRIEVAL_SCORE=0.35
MIN_GROUNDEDNESS_SCORE=0.70

# Source PDF
PDF_URL=https://konverge.ai/pdf/Ebook-Agentic-AI.pdf
PDF_PATH=data/Ebook-Agentic-AI.pdf
```

> **Note**: A template is provided in `.env.example`.

---

## 3. Step-by-Step Execution Commands

### Step 1: Create & Activate Python Virtual Environment

```bash
# Navigate to the project root directory
cd /path/to/agentic_ai_rag_chatbot

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

---

### Step 2: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Run Document Ingestion (Download, Chunk & Index)

This command downloads the eBook (`Ebook-Agentic-AI.pdf`), extracts the text across 60 pages, splits it into 136 chunks, generates 4096-dimensional embeddings with `Qwen/Qwen3-Embedding-8B`, creates the Pinecone index if it doesn't exist, and upserts all vectors.

```bash
python -m scripts.ingest
```

**Expected output:**
```text
Extracted 136 text chunks from Ebook-Agentic-AI.pdf.
Generating sample embedding with Qwen/Qwen3-Embedding-8B to determine dimension...
Embedding dimension detected: 4096
Clearing namespace 'agentic-ai-book' before ingestion...
Upserting 136 chunks into Pinecone in 9 batches of 16...
  Batch 1/9 upserted (16 vectors).
  ...
  Batch 9/9 upserted (8 vectors).
Successfully ingested 136 chunks into Pinecone.
Index: agentic-ai-rag
Namespace: agentic-ai-book
Embedding Model: Qwen/Qwen3-Embedding-8B
```

---

### Step 4: Run the FastAPI REST Server

Start the FastAPI application via `uvicorn`:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Redoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### Query via cURL:
Open another terminal (with the virtual environment activated) and run:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the core definition of Agentic AI as outlined in the eBook?"}'
```

#### Out-of-Scope Test via cURL:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

---

### Step 5: Run the Streamlit Interactive Web Interface

In a new terminal window:

```bash
source .venv/bin/activate
streamlit run app/ui.py
```

The browser will open at [http://localhost:8501](http://localhost:8501).

**UI Features:**
- **Chat Interface**: Ask arbitrary questions about the eBook.
- **Validation Presets**: 1-click test buttons for each of the 6 interview assignment questions.
- **Inspect Retrieved Chunks**: View the text chunks retrieved from Pinecone with page references.
- **Confidence Score Metrics**: Real-time display of combined retrieval and groundedness score.
- **Raw JSON Inspector**: View the raw JSON payload matching the interview specification.

---

### Step 6: Run Automated Validation Script (All 6 Queries)

To automatically execute and score all 6 test queries from Section 4 of the assignment:

```bash
python -m scripts.validate_queries
```

This runs:
1. **Definition & Scope**: "What is the core definition of Agentic AI as outlined in the eBook?"
2. **Architecture & Paradigms**: "What are the main architectural components required to build agentic systems?"
3. **Use Cases**: "What real-world industry use cases for Agentic AI are discussed in the eBook?"
4. **Comparison**: "How does Agentic AI differ from traditional generative AI chatbots according to the text?"
5. **Challenges & Limitations**: "What key challenges or limitations of Agentic AI are mentioned in the document?"
6. **Out-of-Scope Groundedness Check**: "What is the capital of France?" (Properly refused)

---

### Step 7: Run Unit Tests

To run the automated test suite with pytest:

```bash
python -m pytest tests/
```

**Expected output:**
```text
tests/test_api.py .....                                                  [100%]
============================== 5 passed ===============================
```

---

## 4. Response Payload Specification

Every response conforms strictly to the schema required by Section 3C:

```json
{
  "query": "What is the core definition of Agentic AI as outlined in the eBook?",
  "final_answer": "Agentic AI refers to autonomous systems that transition AI from reactive tools to proactive collaborators capable of understanding goals, anticipating needs, making decisions, and taking actions independently...",
  "retrieved_context_chunks": [
    "At its core, Agentic AI is about: A Chef: Expertly manages every task... A Coach: Continuously adapts strategies...",
    "Understanding the Shift from Reactive to Proactive Technology..."
  ],
  "confidence_score": 0.89
}
```

---

## 5. Architecture & LangGraph Graph Flow

```text
                  START
                    |
                    v
              [ retrieve ]
         (Pinecone Semantic Search)
                    |
                    v
              [ generate ]
          (Qwen/Qwen3-8B LLM)
                    |
                    v
               [ grade ]
        (Groundedness Evaluation)
             /            \
     (grounded)       (ungrounded & attempt < 1)
         v                  v
       [END]       [ increment_attempt ]
                            |
                            v
                       [ retrieve ]
```

### Hallucination Prevention & Refusal Protocol
1. **Strict Context Adherence**: The system prompt instructs `Qwen/Qwen3-8B` to answer solely from retrieved text and forbid assumptions.
2. **Deterministic Thresholding**: If the highest retrieval score is below `MIN_RETRIEVAL_SCORE=0.35`, the system refuses without generating ungrounded claims.
3. **Independent Groundedness Grader**: An LLM grading node reviews the question, generated answer, and context chunks to compute a groundedness score from 0.0 to 1.0.
4. **Confidence Formula**:
   $$\text{confidence} = 0.45 \times \text{retrieval\_score} + 0.55 \times \text{groundedness\_score}$$
   For valid refusals of out-of-scope questions, confidence is set to $0.95$.

---

## 6. Git Submission Checklist

When pushing to your public GitHub repository:

- [x] Codebase implemented without no-code/low-code tools.
- [x] `.env` is listed in `.gitignore` (never commit API keys).
- [x] `README.md` and `RUN_INSTRUCTIONS.md` included.
- [x] LangGraph state graph properly constructed and tested.
- [x] Pinecone vector database integration verified.
- [x] Hugging Face models (`Qwen/Qwen3-8B` and `Qwen/Qwen3-Embedding-8B`) integrated.
- [x] FastAPI REST API and Streamlit Web UI both functional.
- [x] All 6 sample validation queries tested.
- [x] Unit test suite passes.
