from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

st.set_page_config(page_title="Overview · Churn Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Business Overview")
st.markdown("High-level KPIs and customer segment breakdowns.")
st.divider()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df


df = load_data()
churned = df[df["Churn"] == "Yes"]
retained = df[df["Churn"] == "No"]

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers", f"{len(df):,}")
k2.metric("Churned", f"{len(churned):,}", delta=f"{len(churned)/len(df):.1%}", delta_color="inverse")
k3.metric("Retained", f"{len(retained):,}")
k4.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
k5.metric("Monthly Revenue at Risk", f"${churned['MonthlyCharges'].sum():,.0f}")

st.divider()

# ── Row 1: Churn donut + Revenue by contract ──────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Churn vs Retained")
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status"] = counts["Status"].map({"Yes": "Churned", "No": "Retained"})
    fig = px.pie(
        counts, names="Status", values="Count",
        color="Status",
        color_discrete_map={"Churned": "#EF4444", "Retained": "#3B82F6"},
        hole=0.55,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("#### Monthly Revenue at Risk by Contract")
    rev = (
        df[df["Churn"] == "Yes"]
        .groupby("Contract")["MonthlyCharges"]
        .sum()
        .reset_index()
        .sort_values("MonthlyCharges", ascending=False)
    )
    rev.columns = ["Contract", "Revenue at Risk"]
    fig2 = px.bar(
        rev, x="Contract", y="Revenue at Risk",
        color="Revenue at Risk",
        color_continuous_scale="Reds",
        text_auto=".2s",
    )
    fig2.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        xaxis_title=None,
        yaxis_title="Monthly $ at Risk",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Tenure bands + Payment method ─────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown("#### Churn Rate by Tenure Band")
    bins = [0, 12, 24, 36, 48, 60, 73]
    labels = ["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "61–72m"]
    df["Tenure Band"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
    band = (
        df.groupby("Tenure Band", observed=True)
        .apply(lambda x: (x["Churn"] == "Yes").mean(), include_groups=False)
        .reset_index()
    )
    band.columns = ["Tenure Band", "Churn Rate"]
    fig3 = px.bar(
        band, x="Tenure Band", y="Churn Rate",
        text=band["Churn Rate"].apply(lambda v: f"{v:.1%}"),
        color="Churn Rate",
        color_continuous_scale="RdYlGn_r",
    )
    fig3.update_layout(
        coloraxis_showscale=False,
        yaxis_tickformat=".0%",
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        xaxis_title=None,
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Churn Rate by Payment Method")
    pay = (
        df.groupby("PaymentMethod")
        .apply(lambda x: (x["Churn"] == "Yes").mean(), include_groups=False)
        .reset_index()
    )
    pay.columns = ["PaymentMethod", "Churn Rate"]
    pay = pay.sort_values("Churn Rate", ascending=True)
    fig4 = px.bar(
        pay, x="Churn Rate", y="PaymentMethod",
        orientation="h",
        text=pay["Churn Rate"].apply(lambda v: f"{v:.1%}"),
        color="Churn Rate",
        color_continuous_scale="Reds",
    )
    fig4.update_layout(
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        yaxis_title=None,
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption(f"Source: {DATA_PATH.name} · {len(df):,} customers")
