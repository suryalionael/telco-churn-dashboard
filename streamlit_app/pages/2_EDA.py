from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

st.set_page_config(page_title="EDA · Churn Intelligence", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 Exploratory Data Analysis")
st.markdown("Interactive charts — use the filters in the sidebar to drill into segments.")
st.divider()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df


df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    contract_filter = st.multiselect(
        "Contract Type",
        options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist(),
    )
    internet_filter = st.multiselect(
        "Internet Service",
        options=df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist(),
    )
    tenure_range = st.slider("Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

filtered = df[
    df["Contract"].isin(contract_filter)
    & df["InternetService"].isin(internet_filter)
    & df["tenure"].between(tenure_range[0], tenure_range[1])
]

st.caption(f"Showing **{len(filtered):,}** of {len(df):,} customers based on current filters.")

COLOR_MAP = {"Yes": "#EF4444", "No": "#3B82F6"}

# ── 1. Churn distribution ──────────────────────────────────────────────────────
st.markdown("### Churn Distribution")
c1, c2 = st.columns([1, 2])

with c1:
    counts = filtered["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    fig = px.pie(
        counts, names="Churn", values="Count",
        color="Churn", color_discrete_map=COLOR_MAP,
        hole=0.6,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    contract_churn = (
        filtered.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Count")
    )
    fig2 = px.bar(
        contract_churn, x="Contract", y="Count", color="Churn",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"Count": "Customers"},
    )
    fig2.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        legend_title_text="Churn",
        xaxis_title=None,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 2. Tenure vs Churn ────────────────────────────────────────────────────────
st.markdown("### Tenure Distribution by Churn Status")
fig3 = px.histogram(
    filtered, x="tenure", color="Churn",
    barmode="overlay",
    opacity=0.75,
    nbins=36,
    color_discrete_map=COLOR_MAP,
    labels={"tenure": "Tenure (months)", "count": "Customers"},
    marginal="box",
)
fig3.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Churn")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── 3. MonthlyCharges vs Churn ────────────────────────────────────────────────
st.markdown("### Monthly Charges by Churn Status")
c3, c4 = st.columns(2)

with c3:
    fig4 = px.box(
        filtered, x="Churn", y="MonthlyCharges",
        color="Churn",
        color_discrete_map=COLOR_MAP,
        points="outliers",
        labels={"MonthlyCharges": "Monthly Charges ($)"},
    )
    fig4.update_layout(
        height=340,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        xaxis_title=None,
    )
    st.plotly_chart(fig4, use_container_width=True)

with c4:
    fig5 = px.violin(
        filtered, x="Churn", y="MonthlyCharges",
        color="Churn",
        color_discrete_map=COLOR_MAP,
        box=True,
        labels={"MonthlyCharges": "Monthly Charges ($)"},
    )
    fig5.update_layout(
        height=340,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        xaxis_title=None,
    )
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── 4. Internet Service breakdown ─────────────────────────────────────────────
st.markdown("### Internet Service & Senior Citizen Breakdown")
c5, c6 = st.columns(2)

with c5:
    inet = (
        filtered.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Count")
    )
    fig6 = px.bar(
        inet, x="InternetService", y="Count", color="Churn",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        labels={"InternetService": "Internet Service"},
    )
    fig6.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title=None,
        legend_title_text="Churn",
    )
    st.plotly_chart(fig6, use_container_width=True)

with c6:
    senior = filtered.copy()
    senior["SeniorCitizen"] = senior["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    sc = (
        senior.groupby(["SeniorCitizen", "Churn"])
        .size()
        .reset_index(name="Count")
    )
    fig7 = px.bar(
        sc, x="SeniorCitizen", y="Count", color="Churn",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"SeniorCitizen": ""},
    )
    fig7.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title=None,
        legend_title_text="Churn",
    )
    st.plotly_chart(fig7, use_container_width=True)

st.divider()

# ── 5. Scatter: Tenure vs MonthlyCharges ─────────────────────────────────────
st.markdown("### Tenure vs Monthly Charges (coloured by Churn)")
sample = filtered.sample(min(len(filtered), 2000), random_state=42)
fig8 = px.scatter(
    sample, x="tenure", y="MonthlyCharges",
    color="Churn",
    color_discrete_map=COLOR_MAP,
    opacity=0.55,
    labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
    hover_data=["Contract", "InternetService"],
)
fig8.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Churn")
st.plotly_chart(fig8, use_container_width=True)
