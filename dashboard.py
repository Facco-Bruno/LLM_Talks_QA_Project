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
    data = json.load(f)

df = pd.DataFrame(data)
if df.empty:
    st.warning("No feedback entries.")
    st.stop()

# Normalize fields
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
if "rewriting_enabled" not in df.columns:
    df["rewriting_enabled"] = False
if "thumbs_up" not in df.columns:
    df["thumbs_up"] = False

# Sidebar filters
st.sidebar.header("Filters")
use_rewriter = st.sidebar.selectbox("Rewriting enabled?", ["All", "Enabled", "Disabled"])
if use_rewriter == "Enabled":
    df = df[df["rewriting_enabled"] == True]
elif use_rewriter == "Disabled":
    df = df[df["rewriting_enabled"] == False]

st.sidebar.write(f"Total items: {len(df)}")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total feedbacks", len(df))
col2.metric("👍 Positive", int(df["thumbs_up"].sum()))
col3.metric("👎 Negative", int((~df["thumbs_up"]).sum()))

# Thumbs rate by rewriting flag
st.subheader("👍 Rate by Rewriting (Enabled vs Disabled)")
if "rewriting_enabled" in df.columns and not df.empty:
    agg = df.groupby("rewriting_enabled")["thumbs_up"].mean().rename({True: "Enabled", False: "Disabled"})
    st.bar_chart(agg)

# Over time
st.subheader("🗓️ Feedbacks over time")
daily = df.groupby("date").size()
st.line_chart(daily)

# By hour
st.subheader("⏰ Feedbacks by hour of day")
hourly = df.groupby("hour").size()
st.bar_chart(hourly)

# Top questions
st.subheader("❓ Most frequent original questions")
if "original_question" in df.columns:
    st.dataframe(df["original_question"].value_counts().head(15).rename("count"))

# Word cloud (original or rewritten)
st.subheader("☁️ Word Cloud")
cloud_source = st.radio("Use text from:", ["original_question", "rewritten_query"], horizontal=True)
text = " ".join(df[cloud_source].dropna().astype(str).tolist())
wc = WordCloud(width=900, height=400, background_color="white").generate(text)
st.image(wc.to_array(), use_column_width=True)
