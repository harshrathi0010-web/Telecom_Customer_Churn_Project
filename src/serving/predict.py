#
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocess_data import preprocess_data

PROJECT_ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=PROJECT_ROOT/"artifacts"/"churn_pipeline.joblib"
INPUT_PATH=PROJECT_ROOT/"data"/"raw"/"new_customers.csv"
OUTPUT_PATH=PROJECT_ROOT/"data"/"processed"/"churn_predictions.csv"


def main() -> None:
    model=joblib.load(MODEL_PATH)
    raw_df=pd.read_csv(INPUT_PATH)
    raw_df.columns=raw_df.columns.str.strip()

    # Keep customer I for the results , preprocessing remove from their feature.
    id_columns=["customerid","CustomerID","customer_id","Customer_ID"]
    id_column=next((col for col in id_columns if col in raw_df.columns),None)

    if id_column:
        customer_ids=raw_df[id_column].copy()
    else:
        customer_ids=pd.Series(range(1,len(raw_df)+1),name="row_number")

    # New Customer have no customer churn value
    features=preprocess_data(raw_df,require_target=False)
    features=features.drop(columns=["Churn"],errors="ignore")

    # use the same features columns as and during the training.
    expected_columns=list(model.feature_names_in_)
    missing_columns=[col for col in expected_columns if col not in features.columns]

    if missing_columns:
        raise ValueError(f" new customer data is missing:{missing_columns}")
    features=features.reindex(columns=expected_columns)

    predictions=model.predict(features)
    probabilities=model.predict_proba(features)[:,1]

    results=pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "Predicted_Churn": pd.Series(predictions).map({0: "No", 1: "Yes"}),
            "Churn_Probability": probabilities.round(3),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(results.head())


if __name__ == "__main__":
    main()
    


