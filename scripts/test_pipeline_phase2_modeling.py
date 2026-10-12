from pathlib import Path
import sys

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add the project root so imports from src work when this script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.models.train import DATA_PATH, MODEL_PATH
from src.utils.validate_data import validate_telco_data


def main() -> None:
    # Load and validate the same raw data used for training.
    raw_df = load_data(str(DATA_PATH))
    is_valid, errors = validate_telco_data(raw_df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")

    # Apply the same cleaning used by the training function.
    clean_df = preprocess_data(raw_df, target_col="Churn")
    X = clean_df.drop(columns=["Churn"])
    y = clean_df["Churn"].astype(int)

    # Recreate the same held-out split used in train.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=22,
        stratify=y,
    )

    # Load the saved Logistic Regression pipeline.
    model = joblib.load(MODEL_PATH)

    # Evaluate the saved model on the held-out rows.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()