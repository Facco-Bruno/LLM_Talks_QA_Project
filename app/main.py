import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# ======= LOAD ENVIRONMENT VARIABLES =======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ======= CONFIGURATION =======
DB_PATH = "models"
FEEDBACK_LOG = "logs/feedback.json"

# ======= LOADER =======
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectordb.as_retriever()

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
def create_qa_chain():
    retriever = load_retriever()
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)
    return chain

# ======= SAVE FEEDBACK =======
def save_feedback(question, response, thumbs_up):
    feedback = {
        "question": question,
        "response": response,
        "thumbs_up": thumbs_up,
        "timestamp": datetime.now().isoformat()
    }
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(feedback)
    with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ======= STREAMLIT INTERFACE =======
def main():
    st.set_page_config(page_title="DataTalks QA", page_icon="💬")
    st.title("💬 Ask about episodes of the DataTalks Club Podcast")

    user_question = st.text_input("Type your question about the episodes:")

    if user_question:
        with st.spinner("🔍 Searching for answer..."):
            qa_chain = create_qa_chain()
            result = qa_chain(user_question)
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown("### 🤖 Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(sources):
            st.markdown(f"**Source {i+1}**")
            st.text(doc.page_content[:500] + "...")

        st.markdown("### 📝 Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                save_feedback(user_question, answer, thumbs_up=True)
                st.success("Feedback saved. Thank you!")
        with col2:
            if st.button("👎 No"):
                save_feedback(user_question, answer, thumbs_up=False)
                st.info("Feedback saved. We'll try to improve!")

if __name__ == "__main__":
    main()
