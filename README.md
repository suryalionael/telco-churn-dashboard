# 📡 Churn Intelligence Platform

> A production-grade machine learning dashboard for predicting and analysing customer churn in the telecommunications industry.

---

## Business Problem

Customer churn costs the telecom industry billions annually. Acquiring a new customer costs 5–7× more than retaining an existing one. This platform combines **exploratory analytics**, **machine learning**, and **actionable business intelligence** to identify at-risk customers before they leave — enabling targeted, cost-effective retention campaigns.

---

## Features

| Feature | Description |
|---|---|
| 📊 **Business Overview** | KPI dashboard with churn rate, revenue at risk, and segment breakdowns |
| 🔍 **Interactive EDA** | Filterable Plotly charts across tenure, charges, contracts, and demographics |
| 🤖 **Live ML Prediction** | Enter a customer profile → instant churn probability with gauge visualisation |
| 💡 **Business Insights** | Data-driven findings and prioritised retention recommendations |

---

## Tech Stack

- **Frontend:** Streamlit (multi-page app)
- **ML:** scikit-learn — Gradient Boosting Classifier inside a full `Pipeline`
- **Visualisation:** Plotly Express / Graph Objects
- **Data:** pandas, numpy
- **Persistence:** joblib

---

## Project Structure

```
Telco_Churn_Project/
├── data/
│   └── raw/
│       └── churn.csv              # IBM Telco dataset (7,043 customers)
├── outputs/
│   └── model/
│       └── churn_model.pkl        # Trained sklearn Pipeline
├── streamlit_app/
│   ├── app.py                     # Main landing page
│   └── pages/
│       ├── 1_Overview.py          # KPI dashboard
│       ├── 2_EDA.py               # Exploratory analysis
│       ├── 3_Prediction.py        # Live churn scoring
│       └── 4_Insights.py          # Business intelligence
├── src/
│   ├── preprocessing.py           # Feature definitions & sklearn transformers
│   └── train_model.py             # Model training script
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd Telco_Churn_Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

Expected output:
```
Loading data ...
Training GradientBoostingClassifier ...

=== Model Performance ===
              precision    recall  f1-score   support
    No Churn       0.85      0.92      0.88      1035
       Churn       0.73      0.56      0.63       374

ROC-AUC: 0.8600+

Model saved → outputs/model/churn_model.pkl
```

### 3. Launch the dashboard

```bash
streamlit run streamlit_app/app.py
```

Then open **http://localhost:8501** in your browser.

---

## Model Details

| Attribute | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Features | tenure, MonthlyCharges, SeniorCitizen, Contract, InternetService, PaymentMethod, Partner, Dependents, PaperlessBilling |
| Preprocessing | StandardScaler (numeric) + OneHotEncoder (categorical) inside ColumnTransformer |
| Output | Binary classification + `predict_proba` churn score |
| Saved format | `joblib` Pipeline (preprocessing + model in one object) |

---

## Dashboard Screenshots

### Home Page
Hero banner with live KPI snapshot — total customers, churn rate, average charges, and monthly revenue at risk.

### Business Overview
Four Plotly charts: churn donut, revenue at risk by contract, churn rate by tenure band, and churn rate by payment method.

### EDA
Seven interactive charts with sidebar filters for contract type, internet service, and tenure range. Includes histograms with marginal box plots, violin plots, and a scatter plot coloured by churn status.

### Prediction
Customer profile form → live Gradient Boosting prediction → gauge chart + risk band classification + probability progress bar.

### Insights
Six key findings, six prioritised recommendations, and a segment churn rate comparison chart with an estimated annual retention value calculation.

---

## Business Value

- **Proactive retention:** Flag at-risk customers before they cancel.
- **Resource prioritisation:** Focus human intervention on high-probability churners (≥70%).
- **Contract strategy:** Data shows incentivising annual contracts reduces churn 3×.
- **Revenue protection:** Each 5% churn reduction saves ~$250K+ per year in this dataset.

---

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 20 features. Publicly available on Kaggle.

---

*Built with Streamlit · scikit-learn · Plotly · pandas*
