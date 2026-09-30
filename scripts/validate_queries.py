#!/usr/bin/env python3
"""Validation script to run all 6 sample queries specified in the assignment."""

import json
import time
from app.rag import RAGPipeline


SAMPLE_QUERIES = [
    {
        "category": "1. Definition & Scope",
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
    },
    {
        "category": "2. Architecture & Paradigms",
        "query": "What are the main architectural components required to build agentic systems?",
    },
    {
        "category": "3. Use Cases",
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    },
    {
        "category": "4. Comparison",
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    },
    {
        "category": "5. Challenges & Considerations",
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
    },
    {
        "category": "6. Out-of-Scope Test (Groundedness Check)",
        "query": "What is the capital of France?",
    },
]


def run_validation():
    print("=" * 80)
    print("Agentic AI eBook RAG Chatbot — Assignment Query Validation")
    print("=" * 80)

    pipeline = RAGPipeline()
    results = []

    for item in SAMPLE_QUERIES:
        category = item["category"]
        query = item["query"]

        print(f"\n▶ Testing {category}")
        print(f"  Query: \"{query}\"")

        output = pipeline.ask(query)
        results.append({"category": category, "result": output})

        print(f"  Answer: {output['final_answer'][:200]}...")
        print(f"  Chunks Retrieved: {len(output['retrieved_context_chunks'])}")
        print(f"  Confidence Score: {output['confidence_score']}")
        time.sleep(1.5)

    print("\n" + "=" * 80)
    print("Summary of Results (Structured JSON format):")
    print("=" * 80)
    print(json.dumps([r["result"] for r in results], indent=2))


if __name__ == "__main__":
    run_validation()
