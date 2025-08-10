import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Hybrid search imports
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Reranking imports (Cross-Encoder)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# Messages for the rewriter
from langchain_core.messages import SystemMessage, HumanMessage

# ======= LOAD ENVIRONMENT VARIABLES =======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ======= CONFIGURATION =======
DB_PATH = "models"
DATA_RAW_DIR = "data/raw"
FEEDBACK_LOG = "logs/feedback.json"

# ======= HELPERS =======
@st.cache_resource
def _load_documents_for_bm25():
    """Load raw TXT docs as LangChain Documents for BM25 keyword retrieval."""
    docs = []
    if not os.path.isdir(DATA_RAW_DIR):
        return docs
    for fname in os.listdir(DATA_RAW_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_RAW_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        docs.append(Document(page_content=content, metadata={"source": fname}))
            except Exception:
                continue
    return docs

# ======= LOADER (HYBRID + RERANK) =======
@st.cache_resource
def load_retriever():
    # 1) Vector retriever (FAISS + best embedding from your eval)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    # 2) Keyword retriever (BM25)
    bm25_docs = _load_documents_for_bm25()
    keyword_retriever = BM25Retriever.from_documents(bm25_docs) if bm25_docs else None
    if keyword_retriever:
        keyword_retriever.k = 8

    # 3) Ensemble (Hybrid Search)
    base_retriever = (
        EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.7, 0.3],
        )
        if keyword_retriever
        else vector_retriever
    )

    # 4) Cross-Encoder Re-ranking (compress to top_n)
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # 5) ContextualCompressionRetriever applies reranking on top of hybrid
    rerank_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    return rerank_retriever

# ======= QUERY REWRITER =======
@st.cache_resource
def _load_rewriter_llm():
    return ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )

def rewrite_query(user_query: str) -> str:
    """
    Rewrite the user's query into a concise, English, retrieval-friendly query.
    Preserve key entities, expand acronyms, add synonyms when helpful, and remove filler.
    Output ONLY the rewritten query.
    """
    if not user_query or not user_query.strip():
        return user_query

    llm = _load_rewriter_llm()
    system = (
        "You are a query rewriting assistant for a RAG system over English podcast transcripts. "
        "Rewrite the user's question into a single concise English search query that maximizes "
        "document retrieval. Preserve named entities and important terms, expand acronyms, add "
        "synonyms when helpful, and remove filler words. Output ONLY the rewritten query."
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_query)])
    rewritten = (resp.content or "").strip()
    return rewritten if rewritten else user_query

# ======= PROMPT =======
template = """
You are a data assistant. Answer the user's question based on the context provided below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate.from_template(template)

# ======= QA CHAIN =======
def create_qa_chain(retriever):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return chain

# ======= SAVE FEEDBACK (now logs rewritten_query + flag + sources) =======
def save_feedback(original_question, rewritten_query, response, thumbs_up, sources, rewriting_enabled):
    record = {
        "original_question": original_question,
        "rewritten_query": rewritten_query,
        "rewriting_enabled": bool(rewriting_enabled),
        "response": response,
        "sources": [s.metadata.get("source", "unknown") for s in (sources or [])],
        "timestamp": datetime.now().isoformat(),
        "thumbs_up": thumbs_up
    }

    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(record)
    with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ======= STREAMLIT INTERFACE =======
def main():
    st.set_page_config(page_title="DataTalks QA", page_icon="💬")
    st.title("💬 Ask about episodes of the DataTalks Club Podcast")

    st.sidebar.header("Retrieval Settings")
    enable_rewrite = st.sidebar.checkbox("Enable Query Rewriting", value=True)
    st.sidebar.caption("Rewrites your question into a concise, retrieval-friendly query (English).")

    user_question = st.text_input("Type your question about the episodes (in English):")

    if user_question:
        with st.spinner("🔍 Searching for answer..."):
            retriever = load_retriever()
            qa_chain = create_qa_chain(retriever)

            # Rewrite query if enabled
            final_query = rewrite_query(user_question) if enable_rewrite else user_question

            # Execute QA with the (possibly rewritten) query
            result = qa_chain(final_query)
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown("### 🤖 Answer")
        st.write(answer)

        with st.expander("🛠️ Retrieval details"):
            st.markdown(f"**Original question:** {user_question}")
            st.markdown(f"**Rewritten query:** {final_query if enable_rewrite else '— (disabled)'}")

        st.markdown("### 📚 Sources")
        if sources:
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}** — {doc.metadata.get('source', 'unknown')}")
                st.text(doc.page_content[:500] + "...")
        else:
            st.info("No sources returned by the retriever.")

        st.markdown("### 📝 Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                save_feedback(
                    original_question=user_question,
                    rewritten_query=final_query if enable_rewrite else user_question,
                    response=answer,
                    thumbs_up=True,
                    sources=sources,
                    rewriting_enabled=enable_rewrite
                )
                st.success("Feedback saved. Thank you!")
        with col2:
            if st.button("👎 No"):
                save_feedback(
                    original_question=user_question,
                    rewritten_query=final_query if enable_rewrite else user_question,
                    response=answer,
                    thumbs_up=False,
                    sources=sources,
                    rewriting_enabled=enable_rewrite
                )
                st.info("Feedback saved. We'll try to improve!")

if __name__ == "__main__":
    main()
