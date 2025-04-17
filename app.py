#!/usr/bin/env python3
# app.py — Streamlit front‑end with hover labels

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Helper: Interquartile range
# ------------------------------------------------------------------
def iqr(x: pd.Series) -> float:
    return x.quantile(0.75) - x.quantile(0.25)

# ------------------------------------------------------------------
# Load precomputed long data
# ------------------------------------------------------------------
@st.cache_data
def load_long(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "df_long.parquet"
    if not path.exists():
        st.error(f"No df_long.parquet in {data_dir}")
        st.stop()
    return pd.read_parquet(path)

# ------------------------------------------------------------------
# Load domain mapping
# ------------------------------------------------------------------
@st.cache_data
def load_domain_map(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "domain_map.csv"
    if not path.exists():
        st.error(f"No domain_map.csv in {base_dir}")
        st.stop()
    # semicolon‐delimited with columns domainId;domain_name
    df = pd.read_csv(path, sep=';')
    # ensure domainId is int
    df['domainId'] = df['domainId'].astype(int)
    return df

# ------------------------------------------------------------------
# Load question mapping
# ------------------------------------------------------------------
@st.cache_data
def load_question_map(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "mapping_file.csv"
    if not path.exists():
        st.error(f"No mapping_file.csv in {base_dir}")
        st.stop()
    df = pd.read_csv(path)  # comma‐delimited
    # rename question column to question_text
    df = df.rename(columns={'question': 'question_text'})
    return df[['question_code','question_text']]

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Radar Charts")
st.title("Interactieve Radar Charts")

# ------------------------------------------------------------------
# Paths and data load
# ------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"

df_long      = load_long(DATA)
domain_map   = load_domain_map(BASE)
question_map = load_question_map(BASE)

# Create lookup dicts
domain_labels   = dict(zip(domain_map['domainId'],   domain_map['domain_name']))
question_labels = dict(zip(question_map['question_code'],
                           question_map['question_text']))

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.header("Globale Filters")

teams = sorted(df_long['team'].unique())
selected_teams = st.sidebar.multiselect("Geïncludeerde Teams:", teams, default=teams)

stat_choice = st.sidebar.selectbox(
    "Statistiek:",
    ["Mediaan (IQR)", "Gemiddelde (±1σ)"]
)

# ------------------------------------------------------------------
# Filtered data
# ------------------------------------------------------------------
df = df_long[df_long['team'].isin(selected_teams)]
domains = sorted(df['domainId'].dropna().unique())

# ------------------------------------------------------------------
# 1) Domain‑level radar
# ------------------------------------------------------------------
st.subheader("1) Overzicht")

if stat_choice == "Mediaan (IQR)":
    agg = (
        df.groupby("domainId")['score']
          .agg(median="median", IQR=iqr)
          .reindex(domains)
          .reset_index()
    )
    center = agg['median']
    spread = agg['IQR']
    title  = "Domein Radar — Mediaan (IQR)"
else:
    agg = (
        df.groupby("domainId")['score']
          .agg(mean="mean", std="std")
          .reindex(domains)
          .reset_index()
    )
    center = agg['mean']
    spread = agg['std']
    title  = "Domein Radar — Gemiddelde (±1σ)"

# Build closed loops
theta        = [f"D{d}" for d in domains]
labels_dom   = [domain_labels[d] for d in domains]
theta_loop   = theta + [theta[0]]
r_lower_loop = list((center - spread).clip(lower=0)) + [(center - spread).iloc[0]]
r_upper_loop = list((center + spread).clip(lower=0)) + [(center + spread).iloc[0]]
r_center_loop= list(center) + [center.iloc[0]]
labels_dom_loop = labels_dom + [labels_dom[0]]

fig_dom = go.Figure()
# invisible lower
fig_dom.add_trace(go.Scatterpolar(
    r=r_lower_loop, theta=theta_loop,
    mode="lines", line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="none", showlegend=False
))
# shaded upper
fig_dom.add_trace(go.Scatterpolar(
    r=r_upper_loop, theta=theta_loop,
    mode="lines", fill="tonext",
    fillcolor="rgba(31,119,180,0.2)",
    hoverinfo="none", showlegend=False
))
# central red line with hover labels
fig_dom.add_trace(go.Scatterpolar(
    r=r_center_loop, theta=theta_loop,
    mode="lines+markers",
    line=dict(color="red", width=2),
    marker=dict(color="red"),
    name="Provincie",
    text=labels_dom_loop,
    hovertemplate="%{text}<br>Score: %{r:.2f}<extra></extra>"
))



fig_dom.update_layout(
    title=title,
    polar=dict(
        angularaxis=dict(
            direction='clockwise',  # ← draw θ in a clockwise order
            rotation=90             # ← optional: start at the top (12 o’cl)
        ),
        radialaxis=dict(range=[1,10], visible=True, tickvals=list(range(1,11)))
    ),
    margin=dict(l=40, r=40, t=80, b=40)
)
st.plotly_chart(fig_dom, use_container_width=True)

# ------------------------------------------------------------------
# 2) Question‑level radar (with friendly domain names)
# ------------------------------------------------------------------
st.subheader("2) Domein Radar")

# Build lists of domain names and a reverse lookup
domain_names   = [domain_labels[d] for d in domains]
name_to_id     = {domain_labels[d]: d for d in domains}

# Show names in the selectbox
selected_name  = st.selectbox("Selecteer een domein:", domain_names)
selected_domain = name_to_id[selected_name]

# Filter for that domain ID
df_q = df[df['domainId'] == selected_domain]
questions = sorted(df_q['question_code'].unique())

if stat_choice == "Mediaan (IQR)":
    agg_q = (
        df_q.groupby("question_code")['score']
            .agg(median="median", IQR=iqr)
            .reindex(questions)
            .reset_index()
    )
    center_q = agg_q['median']
    spread_q = agg_q['IQR']
    title_q  = f"Domain {selected_domain} — Mediaan (IQR)"
else:
    agg_q = (
        df_q.groupby("question_code")['score']
            .agg(mean="mean", std="std")
            .reindex(questions)
            .reset_index()
    )
    center_q = agg_q['mean']
    spread_q = agg_q['std']
    title_q  = f"Domain {selected_domain} — Gemiddelde (±1σ)"

theta_q        = questions
labels_q       = [question_labels[q] for q in questions]
theta_q_loop   = theta_q + [theta_q[0]]
r_lo_q_loop    = list((center_q - spread_q).clip(lower=0)) + [((center_q-spread_q).iloc[0])]
r_hi_q_loop    = list((center_q + spread_q).clip(lower=0)) + [((center_q+spread_q).iloc[0])]
r_center_q_loop= list(center_q) + [center_q.iloc[0]]
labels_q_loop  = labels_q + [labels_q[0]]

fig_q = go.Figure()
fig_q.add_trace(go.Scatterpolar(
    r=r_lo_q_loop, theta=theta_q_loop,
    mode="lines", line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="none", showlegend=False
))
fig_q.add_trace(go.Scatterpolar(
    r=r_hi_q_loop, theta=theta_q_loop,
    mode="lines", fill="tonext",
    fillcolor="rgba(31,119,180,0.2)",
    hoverinfo="none", showlegend=False
))
fig_q.add_trace(go.Scatterpolar(
    r=r_center_q_loop, theta=theta_q_loop,
    mode="lines+markers",
    line=dict(color="red", width=2),
    marker=dict(color="red"),
    name="Provincie",
    text=labels_q_loop,
    hovertemplate="%{text}<br>Score: %{r:.2f}<extra></extra>"
))

fig_q.update_layout(
    title=title_q,
    polar=dict(radialaxis=dict(range=[1,10], visible=True, tickvals=list(range(1,11)))),
    margin=dict(l=40, r=40, t=80, b=40)
)
st.plotly_chart(fig_q, use_container_width=True)