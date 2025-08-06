import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_transcripts(data_path="data/raw", out_path="models/"):
    print("📥 Reading transcript files...")

    docs = []
    for fname in os.listdir(data_path):
        if fname.endswith(".txt"):
            with open(os.path.join(data_path, fname), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(text)

    print(f"✅ {len(docs)} files loaded.")

    print("🔪 Splitting texts into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))

    print(f"🧩 Total chunks created: {len(chunks)}")

    print("📐 Generating embeddings using HuggingFace model...")
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embedding=model)

    os.makedirs(out_path, exist_ok=True)
    vectordb.save_local(out_path)
    print(f"💾 Vector store saved at: {out_path}")

if __name__ == "__main__":
    ingest_transcripts()
