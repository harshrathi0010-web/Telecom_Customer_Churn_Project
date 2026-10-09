import great_expectations as gx
import pandas as pd


def validate_telco_data(df:pd.DataFrame) -> tuple[bool, list[str]]:
    required_columns= [
        "gender",
        "Partner",
        "Dependents",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PhoneService",
        "Churn",
        
    ]

    """
        Comprehensive data validation for the telco customer churn dataset using great expectations.
        
        This function implment critical data quality checks that must pass before model training.
        It validates data integrity,business logic constraints and stat properties.
        that the ML model expect.

        """

    missing=[col for col in required_columns if col not in df.columns]
    if missing:
        return False,[f"Missing required columns:{missing}"]
    errors=[]

    check_df=df.copy() # avoid changing the original data frame



    # Normalize text values for checking
    allowed_values={
        "gender":{"Female","Male"},
        "Partner":{"Yes","No"},
        "Dependents":{"Yes","No"},
        "PhoneService":{"No","Yes"},
        "Churn":{"No","Yes"},
        "InternetService":{"DSL","Fiber optic","No"},
        "Contract": {"Month-to-month", "One year", "Two year"},
    }

    for column in allowed_values:
        check_df[column]=check_df[column].astype("string").str.strip()

    # Convert numeric field for the range expectations.
    for column in ["tenure","MonthlyCharges"]:
        check_df[column]=pd.to_numeric(check_df[column],errors="coerce")

    # Total Charges may be blank in the raw CSV ,allow blanks,reject invalid texts.
    total_text=df["TotalCharges"].astype("string").str.strip()
    total_values=pd.to_numeric(total_text.mask(total_text==""),
                               errors="coerce")

    
    invalid_total = total_text.notna() & total_text.ne("") & total_values.isna()

    if invalid_total.any():
        errors.append("TotalCharges contains non-numeric values.")
    if (total_values.dropna() < 0).any():
        errors.append("TotalCharges contains negative values.")


    # Create a temporary GX context and connect the DataFrame.
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="telco_data")
    asset = source.add_dataframe_asset(name="customers")
    batch_definition = asset.add_batch_definition_whole_dataframe("all_rows")
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": check_df}
    )

    checks = []

    for column in ["Churn", "tenure", "MonthlyCharges"]:
        checks.append(
            (
                f"{column} has no missing values",
                gx.expectations.ExpectColumnValuesToNotBeNull(column=column),
            )
        )

    for column, allowed in allowed_values.items():
        checks.append(
            (
                f"{column} contains valid values",
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=column,
                    value_set=allowed,
                ),
            )
        )

    checks.extend(
        [
            (
                "tenure is between 0 and 120",
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="tenure",
                    min_value=0,
                    max_value=120,
                ),
            ),
            (
                "MonthlyCharges is between 0 and 200",
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="MonthlyCharges",
                    min_value=0,
                    max_value=200,
                ),
            ),
        ]
    )

    for description, expectation in checks:
        result = batch.validate(expectation)
        if not result.success:
            errors.append(description)

    return len(errors) == 0, errors





    

    