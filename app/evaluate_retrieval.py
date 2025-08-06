import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
from typing import List
from tqdm import tqdm

# ===== CONFIG =====
RAW_DATA_PATH = "data/raw"
RESULTS_DIR = "retrieval_evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== FIXED QUESTIONS =====
EVAL_QUESTIONS = [
    {
        "question": "What is the role of a Chief Data Officer?",
        "keywords": ["chief data officer", "responsibility", "cdos"]
    },
    {
        "question": "How can mentoring help in a data career?",
        "keywords": ["mentoring", "mentor", "career growth"]
    },
    {
        "question": "What is the CRISP-DM process?",
        "keywords": ["crisp-dm", "process", "data science"]
    },
    {
        "question": "How to stand out as a data scientist?",
        "keywords": ["stand out", "visibility", "skills", "data scientist"]
    }
]

# ===== STRATEGIES TO TEST =====
CHUNK_CONFIGS = [
    {"chunk_size": 300, "chunk_overlap": 50},
    {"chunk_size": 500, "chunk_overlap": 100},
]

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]


def read_all_documents() -> List[str]:
    documents = []
    for fname in os.listdir(RAW_DATA_PATH):
        if fname.endswith(".txt"):
            with open(os.path.join(RAW_DATA_PATH, fname), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents


def evaluate_retrieval(chunk_size, chunk_overlap, embedding_model_name):
    docs = read_all_documents()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))

    # Embed and index
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = FAISS.from_texts(chunks, embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Evaluation
    results = []
    for item in EVAL_QUESTIONS:
        query = item["question"]
        keywords = item["keywords"]

        retrieved_docs = retriever.get_relevant_documents(query)
        all_text = " ".join([doc.page_content.lower() for doc in retrieved_docs])

        match_score = sum(1 for kw in keywords if kw.lower() in all_text)

        results.append({
            "query": query,
            "matched_keywords": match_score,
            "total_keywords": len(keywords),
            "score": round(match_score / len(keywords), 2)
        })

    return results


def main():
    all_results = []

    for model in EMBEDDING_MODELS:
        for cfg in CHUNK_CONFIGS:
            print(f"🔍 Evaluating: {model} | Chunk: {cfg['chunk_size']} / {cfg['chunk_overlap']}")
            results = evaluate_retrieval(cfg["chunk_size"], cfg["chunk_overlap"], model)
            for res in results:
                res["embedding_model"] = model
                res["chunk_size"] = cfg["chunk_size"]
                res["chunk_overlap"] = cfg["chunk_overlap"]
                all_results.append(res)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "retrieval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    print(df.groupby(["embedding_model", "chunk_size", "chunk_overlap"])["score"].mean().reset_index())


if __name__ == "__main__":
    load_dotenv()
    main()
