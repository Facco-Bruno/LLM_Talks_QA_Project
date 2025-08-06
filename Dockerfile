# ---- Base image ----
FROM python:3.11-slim

# ---- Working directory ----
WORKDIR /app

# ---- Install OS deps ----
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy project files ----
COPY . .

# ---- Install Python deps ----
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- Streamlit config ----
ENV STREAMLIT_PORT=8501
EXPOSE 8501

# ---- Default command (Streamlit UI) ----
CMD ["streamlit", "run", "app/main.py"]
