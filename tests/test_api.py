from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_schema(monkeypatch):
    class FakePipeline:
        def ask(self, query):
            return {
                "query": query,
                "final_answer": "Agentic AI refers to autonomous systems capable of perception, reasoning, and action.",
                "retrieved_context_chunks": [
                    "Agentic AI provides a practical framework for leveraging these systems...",
                    "At its core, Agentic AI is about autonomous goal execution...",
                ],
                "confidence_score": 0.92,
            }

    monkeypatch.setattr("app.main._pipeline", FakePipeline())

    response = client.post(
        "/query",
        json={"query": "What is Agentic AI?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "query",
        "final_answer",
        "retrieved_context_chunks",
        "confidence_score",
    }
    assert data["query"] == "What is Agentic AI?"
    assert isinstance(data["retrieved_context_chunks"], list)
    assert isinstance(data["confidence_score"], float)
    assert data["confidence_score"] == 0.92


def test_out_of_scope_query(monkeypatch):
    class FakePipeline:
        def ask(self, query):
            return {
                "query": query,
                "final_answer": "I’m sorry, but that information is not available in the provided Agentic AI eBook context.",
                "retrieved_context_chunks": [],
                "confidence_score": 0.95,
            }

    monkeypatch.setattr("app.main._pipeline", FakePipeline())

    response = client.post(
        "/query",
        json={"query": "What is the capital of France?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "not available in the provided Agentic AI eBook context" in data["final_answer"]
    assert data["confidence_score"] >= 0.70


def test_query_validation():
    # Empty query should fail validation
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
