"""Train churn model and save pipeline to outputs/model/churn_model.pkl."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_data,
    build_preprocessor,
    FEATURE_COLUMNS,
    TARGET,
)

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"
MODEL_PATH = ROOT / "outputs" / "model" / "churn_model.pkl"


def train() -> None:
    print("Loading data ...")
    df = load_data(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])

    print("Training GradientBoostingClassifier ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    train()
