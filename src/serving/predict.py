from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocess_data import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_pipeline.joblib"


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_customer(customer_data: dict) -> dict:
    """Predict churn for one customer's details."""

    # Turn the customer's form details into a one-row DataFrame.
    raw_df = pd.DataFrame([customer_data])
    raw_df.columns = raw_df.columns.str.strip()

    # The user is asking us to predict Churn, so don't accept it as an input.
    raw_df = raw_df.drop(columns=["Churn"], errors="ignore")

    # Convert form values for numeric fields to numbers.
    numeric_fields = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    for column in numeric_fields:
        if column in raw_df.columns:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    # Clean the inputs using the same cleaning function as training.
    features = preprocess_data(raw_df, require_target=False)

    model = load_model()

    # Ensure the form supplied every feature the model was trained to use.
    expected_columns = list(model.feature_names_in_)
    missing_columns = [
        column for column in expected_columns if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Customer details are missing fields: {missing_columns}")

    # Put columns in the same order the model saw during training.
    features = features.reindex(columns=expected_columns)

    # Predict the label and estimated churn probability.
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "message": "Likely to churn" if prediction == 1 else "Not likely to churn",
        "churn_probability": round(probability, 3),
    }