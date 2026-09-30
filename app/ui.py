import json
import streamlit as st

from app.config import get_settings
from app.rag import RAGPipeline

st.set_page_config(
    page_title="Agentic AI eBook RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .chunk-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


settings = get_settings()
pipeline = load_pipeline()

# Sidebar
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600&auto=format&fit=crop&q=80",
        use_container_width=True,
    )
    st.title("⚙️ System Architecture")
    st.markdown(
        f"""
        **Orchestration**: LangGraph StateGraph  
        **LLM**: `{settings.hf_model}`  
        **Embedding**: `{settings.hf_embedding_model}`  
        **Vector DB**: Pinecone (Serverless)  
        **Index**: `{settings.pinecone_index_name}`  
        **Namespace**: `{settings.pinecone_namespace}`  
        **Dimensions**: 4096  
        """
    )

    st.divider()
    st.subheader("🧪 Validation Query Presets")
    st.caption("Click any sample query to test:")

    sample_queries = [
        "What is the core definition of Agentic AI as outlined in the eBook?",
        "What are the main architectural components required to build agentic systems?",
        "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "What is the capital of France?",
    ]

    selected_query = None
    for idx, query_text in enumerate(sample_queries, start=1):
        label = f"Query {idx}: {query_text[:34]}..."
        if st.button(label, key=f"btn_{idx}", use_container_width=True):
            selected_query = query_text

    st.divider()
    st.markdown(
        """
        **Groundedness Guarantee**:
        Answers are strictly constrained to the *Agentic AI eBook*. Any out-of-scope question will be rejected.
        """
    )

# Main Header
st.markdown('<div class="main-header">🤖 Agentic AI eBook RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangGraph &amp; Pinecone Cyclic Graph RAG powered by Hugging Face models</div>',
    unsafe_allow_html=True,
)

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            details = message["details"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Score", f"{details.get('confidence_score', 0):.2f}")
            with col2:
                st.metric("Retrieved Context Chunks", len(details.get("retrieved_context_chunks", [])))

            with st.expander("📚 View Retrieved Context Chunks"):
                for i, chunk in enumerate(details.get("retrieved_context_chunks", []), start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.info(chunk)

            with st.expander("🔍 View Raw JSON Payload"):
                st.json(details)

# Handle input (either from chat input or clicked sample query button)
prompt = st.chat_input("Ask a question about the Agentic AI eBook...")
if selected_query:
    prompt = selected_query

if prompt:
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Executing LangGraph (Retrieve ➔ Generate ➔ Grounding Grade)..."):
            response_payload = pipeline.ask(prompt)

        final_answer = response_payload["final_answer"]
        confidence = response_payload["confidence_score"]
        chunks = response_payload["retrieved_context_chunks"]

        st.markdown(final_answer)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence Score", f"{confidence:.2f}")
        with col2:
            st.metric("Retrieved Context Chunks", len(chunks))

        with st.expander("📚 View Retrieved Context Chunks"):
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.info(chunk)

        with st.expander("🔍 View Raw JSON Payload"):
            st.json(response_payload)

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "details": response_payload,
        }
    )
