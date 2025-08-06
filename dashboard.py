import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

LOG_PATH = "logs/feedback.json"

st.set_page_config(page_title="Feedback Dashboard", layout="wide")
st.title("📊 LLM Talks QA - Feedback Dashboard")

if not os.path.exists(LOG_PATH):
    st.warning("⚠️ No feedback data found yet.")
    st.stop()

with open(LOG_PATH, "r", encoding="utf-8") as f:
    feedback = json.load(f)

df = pd.DataFrame(feedback)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

col1, col2 = st.columns(2)

# === 1. Total feedbacks por dia
with col1:
    st.subheader("🗓️ Total feedbacks over time")
    feedback_counts = df.groupby("date").size()
    st.line_chart(feedback_counts)

# === 2. Thumbs up vs down
with col2:
    st.subheader("👍 Feedback Distribution")
    thumbs = df["thumbs_up"].value_counts().rename({True: "👍 Up", False: "👎 Down"})
    st.bar_chart(thumbs)

# === 3. Distribuição por hora
st.subheader("⏰ Feedbacks by Hour of Day")
hourly = df.groupby("hour").size()
fig, ax = plt.subplots()
sns.barplot(x=hourly.index, y=hourly.values, ax=ax)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Feedback Count")
st.pyplot(fig)

# === 4. Perguntas mais frequentes
st.subheader("❓ Most Frequent Questions")
top_questions = df["question"].value_counts().head(10)
st.dataframe(top_questions.rename("Count"))

# === 5. Nuvem de palavras das perguntas
st.subheader("☁️ Word Cloud of Questions")
text = " ".join(df["question"].tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
