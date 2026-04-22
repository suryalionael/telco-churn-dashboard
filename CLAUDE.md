# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from the project root (`Telco_Churn_Project/`).

```bash
# Train the model (required before running the app for the first time)
python src/train_model.py

# Launch the Streamlit app
streamlit run streamlit_app/app.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

The project has two independent runtimes that share a single contract via `src/preprocessing.py`.

**Training runtime** (`src/`):
- `preprocessing.py` is the single source of truth for `FEATURE_COLUMNS`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, and `TARGET`. It also exposes `load_data()` (handles `TotalCharges` coercion and binary-encodes `Churn`) and `build_preprocessor()` (returns a `ColumnTransformer` with median-impute + StandardScaler for numerics; mode-impute + OneHotEncoder for categoricals).
- `train_model.py` builds a single sklearn `Pipeline(preprocessor → GradientBoostingClassifier)` and saves the entire fitted pipeline to `outputs/model/churn_model.pkl` via joblib. The pipeline handles all preprocessing internally, so callers never need to transform data manually.

**Streamlit runtime** (`streamlit_app/`):
- `app.py` is the landing page only — it does not use the model.
- Each page in `pages/` is an independent script. Pages resolve file paths using `Path(__file__).parent.parent.parent` (pages → streamlit_app → project root) to reach `data/` and `outputs/`.
- `3_Prediction.py` inserts `ROOT / "src"` onto `sys.path` at import time so it can import `FEATURE_COLUMNS` from `preprocessing.py`. The model is loaded via `@st.cache_resource`; data is loaded via `@st.cache_data`.

**Critical constraint:** The 9 input fields on the Prediction page (`tenure`, `MonthlyCharges`, `SeniorCitizen`, `Contract`, `InternetService`, `PaymentMethod`, `Partner`, `Dependents`, `PaperlessBilling`) must exactly match `FEATURE_COLUMNS` in `preprocessing.py`. If you add, rename, or remove features, you must retrain the model and update the Prediction page form simultaneously.

## Feature & Model Notes

- `TotalCharges` is stored as `object` in the CSV (space strings for new customers); `load_data()` coerces it and drops the 11 affected rows — do not use it as a feature.
- `SeniorCitizen` is already `int` (0/1) in the raw CSV; it does not need encoding.
- The churn class is imbalanced (~26% positive). The model does not use class weighting — adjust `GradientBoostingClassifier` or add `class_weight` handling in `train_model.py` if recall on the churn class needs improvement.
- Risk bands on the Prediction page: Low < 40%, Medium 40–70%, High ≥ 70%.
