from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "outputs" / "model" / "churn_model.pkl"

# Ensure src/ is on the path so preprocessing constants are importable
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(page_title="Prediction · Churn Intelligence", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.risk-high { color: #EF4444; font-weight: 700; font-size: 1.6rem; }
.risk-low  { color: #22C55E; font-weight: 700; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 Churn Prediction")
st.markdown("Enter a customer profile below and get an instant churn probability score.")
st.divider()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

if model is None:
    st.error(
        "**Model not found.** Please train the model first:\n\n"
        "```bash\n"
        "cd Telco_Churn_Project\n"
        "python src/train_model.py\n"
        "```"
    )
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("### Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5
        )
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col2:
        st.markdown("**Service Details**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col3:
        st.markdown("**Personal Details**")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

# ── Prediction output ─────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "Partner": partner,
        "Dependents": dependents,
        "PaperlessBilling": paperless,
    }])[FEATURE_COLUMNS]

    churn_prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.divider()
    st.markdown("### Prediction Result")

    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK**")
            st.markdown(f'<p class="risk-high">Churn Probability: {churn_prob:.1%}</p>', unsafe_allow_html=True)
            st.markdown("This customer profile is associated with **high churn likelihood**. Immediate retention action is recommended.")
        else:
            st.success("✅ **LOW CHURN RISK**")
            st.markdown(f'<p class="risk-low">Churn Probability: {churn_prob:.1%}</p>', unsafe_allow_html=True)
            st.markdown("This customer profile indicates **low churn risk**. Continue standard engagement.")

        st.markdown("**Risk Meter**")
        st.progress(float(churn_prob))

        # Confidence bands
        if churn_prob >= 0.7:
            band, band_color = "High Risk (≥70%)", "🔴"
        elif churn_prob >= 0.4:
            band, band_color = "Medium Risk (40–70%)", "🟡"
        else:
            band, band_color = "Low Risk (<40%)", "🟢"

        st.markdown(f"**Risk Band:** {band_color} {band}")

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#EF4444" if prediction == 1 else "#22C55E"},
                "steps": [
                    {"range": [0, 40],  "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF9C3"},
                    {"range": [70, 100],"color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#1F2937", "width": 3},
                    "thickness": 0.75,
                    "value": churn_prob * 100,
                },
            },
            title={"text": "Churn Probability", "font": {"size": 18}},
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Input summary table ────────────────────────────────────────────────────
    with st.expander("View Input Features"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
