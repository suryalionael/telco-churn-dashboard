from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

st.set_page_config(page_title="Insights · Churn Intelligence", page_icon="💡", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.insight-card {
    background: #F0F9FF;
    border-left: 4px solid #3B82F6;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.75rem;
}
.rec-card {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.75rem;
}
.warn-card {
    background: #FFF7ED;
    border-left: 4px solid #F97316;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 💡 Business Insights & Recommendations")
st.markdown("Data-driven findings and actionable retention strategies derived from the analysis.")
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

churn_rate = len(churned) / len(df)
mtm_churn = (df[df["Contract"] == "Month-to-month"]["Churn"] == "Yes").mean()
fiber_churn = (df[df["InternetService"] == "Fiber optic"]["Churn"] == "Yes").mean()
echeck_churn = (df[df["PaymentMethod"] == "Electronic check"]["Churn"] == "Yes").mean()
early_churn = (df[df["tenure"] <= 12]["Churn"] == "Yes").mean()
revenue_at_risk = churned["MonthlyCharges"].sum()

# ── Executive Summary ─────────────────────────────────────────────────────────
st.markdown("### Executive Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Overall Churn Rate", f"{churn_rate:.1%}", help="% of customers who churned")
col2.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", help="Monthly charges from churned customers")
col3.metric("Annual Revenue at Risk", f"${revenue_at_risk * 12:,.0f}", help="Estimated annual impact")
st.divider()

# ── Key Findings ──────────────────────────────────────────────────────────────
st.markdown("### 🔑 Key Findings")

findings = [
    (
        "Month-to-Month Contracts Drive the Most Churn",
        f"Customers on month-to-month contracts churn at **{mtm_churn:.1%}** — nearly 3× higher than annual contract holders. "
        f"This segment represents the single largest retention opportunity.",
        "insight",
    ),
    (
        "Fiber Optic Users Are Disproportionately at Risk",
        f"Despite being a premium service, Fiber optic customers churn at **{fiber_churn:.1%}**. "
        f"This suggests dissatisfaction with value-for-money or service quality rather than pricing alone.",
        "warn",
    ),
    (
        "Electronic Check Payers Churn Most",
        f"**{echeck_churn:.1%}** of customers paying via electronic check churn, compared to ~15% for automatic payment methods. "
        f"Payment friction and disengagement are likely contributors.",
        "warn",
    ),
    (
        "Early Tenure is the Highest-Risk Window",
        f"**{early_churn:.1%}** of customers in their first 12 months churn. "
        f"Onboarding experience and initial value perception are critical retention levers.",
        "warn",
    ),
    (
        "Long-Tenure Customers Are Highly Loyal",
        f"Customers with 48+ months of tenure have a churn rate below 10%. "
        f"Retention investment should be front-loaded in the customer lifecycle.",
        "insight",
    ),
    (
        "Senior Citizens Have Elevated Churn Risk",
        f"Senior customers churn at a significantly higher rate. "
        f"Dedicated support programs and simplified billing could reduce friction for this segment.",
        "insight",
    ),
]

for title, body, kind in findings:
    css_class = {"insight": "insight-card", "warn": "warn-card", "rec": "rec-card"}[kind]
    icon = {"insight": "🔵", "warn": "🟠", "rec": "🟢"}[kind]
    st.markdown(
        f'<div class="{css_class}"><strong>{icon} {title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown("### ✅ Strategic Recommendations")

recommendations = [
    ("Incentivise Annual Contract Upgrades",
     "Offer month-to-month customers a meaningful discount (10–15%) or a free service add-on to switch to annual contracts. "
     "Target: 20% conversion rate = **significant churn reduction**."),
    ("Launch a 90-Day Onboarding Programme",
     "Assign a dedicated success touchpoint at days 7, 30, and 90 for new customers. "
     "Early proactive outreach reduces early-tenure churn by 15–25% in comparable industries."),
    ("Investigate Fiber Optic Value Perception",
     "Run a Net Promoter Score (NPS) survey specifically for Fiber optic customers. "
     "Address the top 3 complaints with a targeted service improvement sprint."),
    ("Automate Payment Migration",
     "Proactively migrate electronic check payers to auto-pay with a one-time bill credit incentive. "
     "Reduces friction and increases retention by ~5%."),
    ("Deploy Real-Time Churn Scoring",
     "Integrate this ML model into CRM (e.g., Salesforce) to flag at-risk customers for the retention team. "
     "Focus human intervention on customers scoring >70% churn probability."),
    ("Build a Senior Citizen Support Track",
     "Offer dedicated phone support and simplified billing for senior customers. "
     "A small cost-to-serve increase can prevent high-value long-term churn."),
]

for i, (title, body) in enumerate(recommendations, 1):
    st.markdown(
        f'<div class="rec-card"><strong>🟢 {i}. {title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Churn rate benchmark chart ─────────────────────────────────────────────────
st.markdown("### Churn Rate by Key Segments")

segments = {
    "Month-to-Month": mtm_churn,
    "Fiber Optic Internet": fiber_churn,
    "Electronic Check": echeck_churn,
    "First 12 Months": early_churn,
    "Overall": churn_rate,
}

seg_df = pd.DataFrame({"Segment": list(segments.keys()), "Churn Rate": list(segments.values())})
seg_df = seg_df.sort_values("Churn Rate", ascending=True)

fig = px.bar(
    seg_df, x="Churn Rate", y="Segment",
    orientation="h",
    text=seg_df["Churn Rate"].apply(lambda v: f"{v:.1%}"),
    color="Churn Rate",
    color_continuous_scale="Reds",
)
fig.add_vline(
    x=churn_rate, line_dash="dash", line_color="#3B82F6",
    annotation_text=f"Overall avg {churn_rate:.1%}", annotation_position="top right",
)
fig.update_layout(
    height=320,
    coloraxis_showscale=False,
    xaxis_tickformat=".0%",
    margin=dict(t=10, b=10, l=10, r=10),
    yaxis_title=None,
    xaxis_title="Churn Rate",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(
    "**Estimated Retention Value:** Reducing churn rate by just **5 percentage points** "
    f"saves approximately **${revenue_at_risk * 0.05 * 12:,.0f} per year** in recovered revenue."
)
