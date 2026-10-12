from pathlib import Path
import sys

import pandas as pd

# Add the project root so imports from src work when this script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.utils.validate_data import validate_telco_data


# Path to the raw CSV in your project.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer_churn_data.csv"


def main() -> None:
    # Load the original customer data.
    raw_df = load_data(str(DATA_PATH))
    print(f"Loaded data: {raw_df.shape}")

    # Check the raw data before cleaning it.
    is_valid, errors = validate_telco_data(raw_df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")
    print("Data validation passed.")

    # Clean the data using the same function as model training.
    clean_df = preprocess_data(raw_df, target_col="Churn")
    print(f"Preprocessed data: {clean_df.shape}")

    # Confirm Churn is present, has no missing values, and contains only 0 or 1.
    assert "Churn" in clean_df.columns, "Churn column is missing."
    assert clean_df["Churn"].notna().all(), "Churn contains missing values."
    assert set(clean_df["Churn"].unique()).issubset({0, 1}), (
        "Churn must contain only 0 and 1."
    )

    # Confirm TotalCharges was converted to a numeric column.
    if "TotalCharges" in clean_df.columns:
        assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"]), (
            "TotalCharges is not numeric."
        )

    print("Phase 1 checks passed.")


if __name__ == "__main__":
    main()