"""
dashboard/app.py — Streamlit dashboard for RAG failure analysis results.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import glob
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Failure Miner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid;
        margin-bottom: 0.5rem;
    }
    .stDataFrame { font-size: 0.85rem; }
    .failure-PASS { color: #00d26a; }
    .failure-HALLUCINATION { color: #ff4b4b; }
    .failure-RETRIEVAL_DRIFT { color: #ffa500; }
    .failure-CONTEXT_DROP { color: #a78bfa; }
    .failure-REASONING_FAILURE { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — load results
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 RAG Failure Miner")
st.sidebar.markdown("---")

results_dir = st.sidebar.text_input("Results directory", value="results")
result_files = sorted(glob.glob(f"{results_dir}/*.json"))

if not result_files:
    st.warning(f"No result files found in `{results_dir}/`. Run `python run_eval.py` first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select evaluation run",
    result_files,
    format_func=lambda x: Path(x).stem,
)

with open(selected_file) as f:
    data = json.load(f)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("RAG Failure Case Mining & Error Analysis")
st.markdown(f"**Run:** `{data['run_name']}` | **Config:** {data['config']}")
st.markdown("---")

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

metrics = data["aggregate_metrics"]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Pass Rate", f"{metrics.get('pass_rate', 0):.1%}")
with col2:
    st.metric("Failure Rate", f"{metrics.get('failure_rate', 0):.1%}")
with col3:
    st.metric("Context Relevance", f"{metrics.get('mean_context_relevance', 0):.3f}")
with col4:
    st.metric("Answer Faithfulness", f"{metrics.get('mean_answer_faithfulness', 0):.3f}")
with col5:
    st.metric("Chunk Precision", f"{metrics.get('mean_chunk_precision', 0):.3f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Failure distribution
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Failure Category Distribution")

    dist = data["failure_distribution"]
    colors = {
        "PASS": "#00d26a",
        "HALLUCINATION": "#ff4b4b",
        "RETRIEVAL_DRIFT": "#ffa500",
        "CONTEXT_DROP": "#a78bfa",
        "REASONING_FAILURE": "#60a5fa",
    }

    fig_pie = px.pie(
        names=list(dist.keys()),
        values=list(dist.values()),
        color=list(dist.keys()),
        color_discrete_map=colors,
        hole=0.4,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend=dict(orientation="v"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Retrieval Quality Scores")

    score_keys = ["mean_context_relevance", "mean_answer_faithfulness",
                  "mean_chunk_precision", "mean_keyword_recall"]
    score_labels = ["Context\nRelevance", "Answer\nFaithfulness",
                    "Chunk\nPrecision", "Keyword\nRecall"]
    score_vals = [metrics.get(k, 0) for k in score_keys]

    fig_bar = go.Figure(go.Bar(
        x=score_labels,
        y=score_vals,
        marker_color=["#60a5fa", "#34d399", "#fbbf24", "#f472b6"],
        text=[f"{v:.3f}" for v in score_vals],
        textposition="outside",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(range=[0, 1.1], gridcolor="#2a2d3e"),
        xaxis=dict(gridcolor="#2a2d3e"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-query results table
# ---------------------------------------------------------------------------

st.subheader("Per-Query Results")

per_query = data["per_query"]
df = pd.DataFrame([
    {
        "ID": q["query_id"],
        "Query": q["query"][:60] + ("..." if len(q["query"]) > 60 else ""),
        "Answer": q["generated_answer"][:70] + ("..." if len(q["generated_answer"]) > 70 else ""),
        "Category": q["failure_category"],
        "Ctx Relevance": q["retrieval_scores"].get("context_relevance", 0),
        "Faithfulness": q["retrieval_scores"].get("answer_faithfulness", 0),
        "Chunk Precision": q["retrieval_scores"].get("chunk_precision", 0),
    }
    for q in per_query
])

# Colour-code failure category
def color_category(val):
    colors_map = {
        "PASS": "color: #00d26a",
        "HALLUCINATION": "color: #ff4b4b",
        "RETRIEVAL_DRIFT": "color: #ffa500",
        "CONTEXT_DROP": "color: #a78bfa",
        "REASONING_FAILURE": "color: #60a5fa",
    }
    return colors_map.get(val, "")

# Filter by category
categories = ["All"] + sorted(df["Category"].unique().tolist())
selected_cat = st.selectbox("Filter by failure category", categories)

if selected_cat != "All":
    df_filtered = df[df["Category"] == selected_cat]
else:
    df_filtered = df

st.dataframe(
    df_filtered.style.applymap(color_category, subset=["Category"]),
    use_container_width=True,
    height=350,
)

# ---------------------------------------------------------------------------
# Deep-dive: individual query inspector
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Query Inspector")

query_ids = [q["query_id"] for q in per_query]
selected_qid = st.selectbox("Select a query to inspect", query_ids)

selected_q = next(q for q in per_query if q["query_id"] == selected_qid)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"**Query:** {selected_q['query']}")
    st.markdown(f"**Ground Truth:** `{selected_q['ground_truth']}`")
    st.markdown(f"**Generated Answer:** {selected_q['generated_answer']}")
    st.markdown(f"**Prompt Variant:** `{selected_q.get('prompt_used', 'N/A')}`")

    cat = selected_q["failure_category"]
    cat_colors = {
        "PASS": "🟢", "HALLUCINATION": "🔴",
        "RETRIEVAL_DRIFT": "🟠", "CONTEXT_DROP": "🟣", "REASONING_FAILURE": "🔵",
    }
    st.markdown(f"**Failure Category:** {cat_colors.get(cat, '⚪')} `{cat}`")
    st.markdown(f"**Judge Reasoning:** _{selected_q.get('judge_reasoning', '')}_")

with c2:
    scores = selected_q["retrieval_scores"]
    if scores:
        fig_radar = go.Figure(go.Scatterpolar(
            r=[
                scores.get("context_relevance", 0),
                scores.get("answer_faithfulness", 0),
                scores.get("chunk_precision", 0),
                scores.get("keyword_recall", 0),
            ],
            theta=["Context\nRelevance", "Answer\nFaithfulness", "Chunk\nPrecision", "Keyword\nRecall"],
            fill="toself",
            line_color="#60a5fa",
            fillcolor="rgba(96,165,250,0.2)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2d3e", color="white"),
                angularaxis=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Multi-run comparison (if multiple files)
# ---------------------------------------------------------------------------

if len(result_files) > 1:
    st.markdown("---")
    st.subheader("Multi-Run Comparison")

    comparison_rows = []
    for f in result_files:
        with open(f) as fp:
            d = json.load(fp)
        row = {"run": Path(f).stem, **d.get("aggregate_metrics", {})}
        comparison_rows.append(row)

    df_comp = pd.DataFrame(comparison_rows).set_index("run")

    metric_to_plot = st.selectbox(
        "Metric to compare",
        ["pass_rate", "mean_context_relevance", "mean_answer_faithfulness",
         "mean_chunk_precision", "mean_keyword_recall"],
    )

    if metric_to_plot in df_comp.columns:
        fig_comp = px.bar(
            df_comp.reset_index(),
            x="run", y=metric_to_plot,
            color="run",
            text=metric_to_plot,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
            yaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.markdown(
    "<center><small>RAG Failure Miner • Built with LangChain, Sentence Transformers, W&B</small></center>",
    unsafe_allow_html=True,
)