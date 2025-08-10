# DataTalksClub Podcast RAG (LLM Zoomcamp Final Project)
===========================================================

# OVERVIEW
--------
An end-to-end Retrieval-Augmented Generation (RAG) application that answers questions about DataTalks.Club podcast episodes using their public transcripts. The project includes an automated ingestion pipeline, hybrid retrieval (vector + keyword), cross-encoder re-ranking, query rewriting, a Streamlit UI, a monitoring dashboard with 5+ charts, evaluation scripts for retrieval and LLM quality, and full containerization via Docker.

# FEATURES
--------
- Automated data ingestion (chunking + embeddings + FAISS).
- Hybrid Search: Vector (FAISS + sentence-transformers) + BM25 keyword search.
- Document Re-ranking: Cross-Encoder (MS MARCO MiniLM) improves contextual relevance.
- Query Rewriting: Reformulates user queries for better retrieval recall (toggle in the UI).
- Streamlit UI: Ask questions and view sources used for answers.
- Feedback Logging: Saves original question, rewritten query, sources, timestamp, and thumbs up/down.
- Monitoring Dashboard: 5+ charts (time series, thumbs ratio, hour-of-day distribution, top questions, word cloud).
- Evaluation: Retrieval strategy comparison + LLM response quality metrics.
- Docker & docker-compose: Reproducible local runs, volumes for data/logs/models.

# REPO STRUCTURE
--------------
.
├── app/
│   ├── main.py                    (Streamlit RAG app: hybrid search + reranking + query rewrite)
│   ├── ingest.py                  (Ingestion pipeline: chunk + embed + FAISS index)
│   ├── evaluate_retrieval.py      (Compares embeddings/chunking configs)
│   ├── evaluate_llm.py            (Compares prompts/models; saves responses)
│   └── evaluate_llm_metrics.py    (BLEU/ROUGE-L/Fuzzy metrics on generated answers)
├── data/
│   └── raw/                       (TXT transcripts downloaded here)
├── evaluation/
│   └── qa_dataset.json            (Small QA set used by evaluate_llm.py)
├── logs/
│   └── feedback.json              (User feedback + rewritten queries + sources)
├── models/                        (FAISS index: index.faiss + index.pkl)
├── dashboard.py                   (Monitoring dashboard: 5+ charts)
├── download_transcripts.py        (Scraper for transcripts from datatalks.club)
├── requirements.txt               (Pinned dependencies for reproducibility)
├── docker-compose.yml             (Main app + dashboard services)
├── Dockerfile                     (Base image + install + default CMD)
├── .env.example                   (Template for secrets)
└── README.txt                     (This file)

# PREREQUISITES
-------------
- Python 3.11+
- OpenAI API key (for ChatOpenAI and query rewriting)
- (Optional) Docker and docker-compose

# QUICKSTART — FULL STEP-BY-STEP (FROM DATA DOWNLOAD)
---------------------------------------------------
This guide walks you from **downloading the raw transcripts** to running the UI and dashboard.

### 0) Create and activate a virtual environment
Windows (PowerShell):
```
python -m venv .venv
. .venv\Scripts\Activate.ps1
```
macOS/Linux:
```
python -m venv .venv
source .venv/bin/activate
```

### 1) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure the OpenAI key
- Copy `.env.example` to `.env` and set your key:
```
OPENAI_API_KEY=sk-...
```

### 3) Download the podcast transcripts (RAW DATA)
Run the scraper (it skips episodes without a transcript and captures the full transcript content):
```
python download_transcripts.py
```
Expected: `.txt` files created under `data/raw/`.  
Tip: If you want to test quickly, you can temporarily limit the number of links inside the script.

### 4) Build the vector index (INGESTION)
Ingestion will:
- Read all `.txt` files in `data/raw/`
- Split into chunks (recommended: size=300, overlap=50)
- Generate embeddings with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- Save FAISS index to `models/`
```
python app/ingest.py
```
Expected: `models/index.faiss` and `models/index.pkl` present.

### 5) Run the RAG application (UI)
```
streamlit run app/main.py
```
Open http://localhost:8501

Notes:
- Ask questions **in English** (the dataset is in English).
- You can toggle **Query Rewriting** in the sidebar.
- The app shows sources used for each answer.

### 6) Provide feedback (Thumbs up/down)
- Each answer page has 👍/👎.
- Feedback is stored in `logs/feedback.json` with:
  - `original_question`, `rewritten_query`, `rewriting_enabled`
  - `sources`, `response`, `thumbs_up`, `timestamp`

### 7) Monitor and analyze (Dashboard with 5+ charts)
```
streamlit run dashboard.py
```
- If running together with the main app via docker-compose, the dashboard maps to port 8502.
- Charts included: time series, thumbs ratio, hour-of-day, top questions, word clouds (original vs rewritten).

### 8) Evaluate retrieval strategies (optional but recommended)
Compare embeddings/chunking configurations and pick the best:
```
python app/evaluate_retrieval.py
```
Output:
- `retrieval_evaluation/retrieval_results.csv`
Recommended (per our sample results):
- Embedding: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- Chunking: `chunk_size=300`, `chunk_overlap=50`

### 9) Evaluate LLM responses (optional but recommended)
Run multiple prompts/models, collect responses:
```
python app/evaluate_llm.py
```
Outputs:
- `evaluation/llm_evaluation_results.csv`
- `evaluation/llm_manual_review.csv` (template for manual scoring)

Compute automatic metrics (BLEU/ROUGE-L/Fuzzy):
```
python app/evaluate_llm_metrics.py
```
Output:
- `evaluation/llm_evaluation_metrics.csv`

SETUP (WITH DOCKER)
-------------------
1) Ensure Docker and docker-compose are installed.

2) Create `.env` with your OpenAI key:
```
OPENAI_API_KEY=sk-...
```

3) Build and run both services (main app + dashboard):
```
docker-compose up --build
```
- Main app:    http://localhost:8501  
- Dashboard:   http://localhost:8502

Volumes persist your local `./data`, `./models`, and `./logs` folders inside the container.

# DATA INGESTION DETAILS
----------------------
- Chunking strategy (based on retrieval evaluation):
  - chunk_size = 300
  - chunk_overlap = 50
- Embeddings (best from evaluation):
  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1
- Vector store:
  - FAISS (models/index.faiss, models/index.pkl)

# RETRIEVAL PIPELINE
------------------
1) Optional Query Rewriting (OpenAI): rewrites user question into a concise English query (toggle in UI).
2) Hybrid Retrieval:
   - Vector search via FAISS (k=8)
   - Keyword search via BM25 (k=8)
   - EnsembleRetriever with weights [0.7 (vector), 0.3 (BM25)]
3) Cross-Encoder Re-ranking:
   - cross-encoder/ms-marco-MiniLM-L-6-v2
   - Top-N reranking (top_n=5)
4) LLM Answering:
   - ChatOpenAI (gpt-4o-mini by default)
   - “Stuff” chain with retrieved contexts

# USER INTERFACE (STREAMLIT)
--------------------------
- Single text input for English questions.
- Displays LLM answer and up to N source excerpts.
- Feedback buttons (👍/👎).
- “Retrieval details” expander shows original vs. rewritten query.

# MONITORING DASHBOARD
--------------------
- Loads `logs/feedback.json` and builds 5+ charts:
  - Feedbacks over time
  - Thumbs up/down distribution
  - Hour-of-day distribution
  - Top original questions
  - Word cloud (original vs rewritten queries)
- Sidebar filter to compare with/without rewriting.

# EVALUATION: RETRIEVAL
---------------------
Run:
```
python app/evaluate_retrieval.py
```
What it does:
- Tests multiple chunking configs and embedding models.
- Scores based on keyword coverage in retrieved docs.
- Saves CSV:
  `retrieval_evaluation/retrieval_results.csv`

Example result (yours may vary):
- `multi-qa-MiniLM-L6-cos-v1` with `chunk_size=300`, `overlap=50` scored best.

# EVALUATION: LLM RESPONSES
-------------------------
1) Prepare a small QA dataset:
   See `evaluation/qa_dataset.json` (example included).

2) Generate answers with different prompts/models:
```
python app/evaluate_llm.py
```
Outputs:
- `evaluation/llm_evaluation_results.csv`
- `evaluation/llm_manual_review.csv` (template for manual scoring)

3) Compute automatic metrics:
```
python app/evaluate_llm_metrics.py
```
Output:
- `evaluation/llm_evaluation_metrics.csv` (BLEU, ROUGE-L, Fuzzy)

# BEST PRACTICES IMPLEMENTED
--------------------------
- Hybrid Search (vector + BM25)  [✓ Best Practices +1]
- Document Re-ranking (Cross-Encoder)  [✓ Best Practices +1]
- Query Rewriting (LLM-aided)  [✓ Best Practices +1]

# PEER REVIEW GUIDE
-----------------
To check the exact code state for a commit:
```
https://github.com/{username}/{repo-name}/tree/{commit-hash}
```
Or clone and reset to the commit:
```
git clone https://github.com/{username}/{repo-name}.git
cd {repo-name}
git reset --hard {commit-hash}
```

# REPRODUCIBILITY
---------------
- Pinned dependencies in `requirements.txt`.
- `.env.example` for secrets.
- Clear step-by-step run instructions (local and Docker).
- Data generated programmatically (`download_transcripts.py`).
- Vector index regenerated via `app/ingest.py`.

# TROUBLESHOOTING
---------------
1) sentence-transformers / transformers / huggingface-hub version issues  
   If you encounter import errors (e.g., `cached_download` removed, or `PreTrainedModel` not found), ensure:
```
pip install -U "transformers<4.40.0" "sentence-transformers==2.2.2" "huggingface-hub==0.15.1"
```

2) Missing OpenAI key  
   Ensure `.env` has `OPENAI_API_KEY` and the app loads it (dotenv).  
   You can also export the var in your shell.

3) Slow first run  
   The first run may download models (sentence-transformers, cross-encoder). Subsequent runs are faster.

4) Empty results  
   Ensure `data/raw` contains TXT transcripts and `models/` has `index.faiss` + `index.pkl` after ingestion.

# SECURITY & PRIVACY
------------------
- The app uses OpenAI API only for query rewriting and answer generation.
- No user secrets are logged. Feedback logs store questions, rewritten queries, sources, and thumbs.

# LICENSE
-------
MIT (feel free to adjust if you prefer another license).

# ACKNOWLEDGEMENTS
----------------
- DataTalks.Club for the podcast and transcripts.
- DataTalksClub ML Zoomcamp team and community.
- OpenAI, HuggingFace, LangChain, FAISS, Streamlit, and related open-source projects.

# CONTACT
-------
For questions about this project: open an issue in the repository or reach out on DataTalksClub Slack.
