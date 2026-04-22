"""Main entry point — run with: streamlit run streamlit_app/app.py"""
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.title("Churn Intelligence")
    st.caption("Telco Customer Analytics · v1.0")
    st.divider()
    st.markdown("""
**Navigate using the pages above:**
- 📊 Overview — KPI dashboard
- 🔍 EDA — Exploratory analysis
- 🤖 Prediction — Live churn score
- 💡 Insights — Business findings
""")
    st.divider()
    st.caption("Dataset: IBM Telco Customer Churn")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Churn Intelligence Platform")
st.markdown(
    "A production-grade analytics dashboard that combines **exploratory analysis**, "
    "**machine learning**, and **business intelligence** to identify customers at risk "
    "of churning — before they leave."
)

st.divider()

# ── Quick snapshot ────────────────────────────────────────────────────────────
@st.cache_data
def load_snapshot():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df

df = load_snapshot()
churn_rate = (df["Churn"] == "Yes").mean()
revenue_at_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churn_rate:.1%} target", delta_color="inverse")
col3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
col4.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

st.divider()

# ── Feature grid ──────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("#### 📊 Business Overview")
    st.markdown("Track churn KPIs, revenue at risk, and segment breakdowns in real time.")
with col_b:
    st.markdown("#### 🔍 Interactive EDA")
    st.markdown("Explore tenure patterns, charge distributions, and contract impact on churn.")
with col_c:
    st.markdown("#### 🤖 ML Prediction")
    st.markdown("Enter a customer profile and get an instant churn probability powered by Gradient Boosting.")

st.divider()
st.caption("Built with Streamlit · scikit-learn · Plotly · pandas")
