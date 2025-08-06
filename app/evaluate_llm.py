import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_PATH = "models"
DATASET_PATH = "evaluation/qa_dataset.json"
OUTPUT_PATH = "evaluation/llm_evaluation_results.csv"

PROMPTS = {
    "simple": """
Answer the following question using the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
""",
    "instructional": """
You are an expert assistant specialized in podcast content. Answer the user's question using only the context below. If the answer is not present, be honest and say you don't know.

Context:
{context}

User's Question:
{question}
"""
}

MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectordb.as_retriever()

def evaluate():
    results = []
    dataset = load_dataset(DATASET_PATH)
    retriever = load_retriever()

    for model_name in MODELS:
        for prompt_name, template in PROMPTS.items():
            prompt = PromptTemplate.from_template(template)
            llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

            for item in dataset:
                try:
                    result = qa_chain.run(item["question"])
                    results.append({
                        "model": model_name,
                        "prompt": prompt_name,
                        "question": item["question"],
                        "expected_answer": item["answer"],
                        "generated_answer": result
                    })
                except Exception as e:
                    print(f"Error with question: {item['question']} — {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    evaluate()
