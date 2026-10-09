# 
import pandas as pd

def preprocess_data(df:pd.DataFrame,target_col:str="Churn") -> pd.DataFrame:

    # Basic cleaning for the telecom churn
    # simple NA or data handling
    # Drop customer ID
    # fix total charges to numeric
    # trim the columns
    # map the churn into 0/1 if needed
    df=df.copy() # avoid changing the original data frame
    

    df.columns=df.columns.str.strip()   # Remove the whitespace and trim the columns

     # Drop ID columns if present.
    id_columns = ["customerID", "CustomerID", "customer_id", "Customer_ID"]
    df = df.drop(
        columns=[col for col in id_columns if col in df.columns]
    )


    if target_col not in df.columns:
        raise ValueError(f"Required target column '{target_col}' is missing.")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        labels = df[target_col].astype("string").str.strip()
        df[target_col] = labels.map({"No": 0, "Yes": 1})

    if df[target_col].isna().any():
        raise ValueError(f"{target_col} has missing or unexpected values.")

    df[target_col] = df[target_col].astype(int)   

    # change the data type of TotalCharges into numeric
    if "TotalCharges" in df.columns:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors="coerce")

    # Senior citizen should be 0/1 int if present.
    if "SeniorCitizen" in df.columns:
        df['SeniorCitizen']=pd.to_numeric(df['SeniorCitizen'],errors="coerce",).fillna(0).astype(int)

    # Simple NA strategy
    # if num col fill with the 0.
    # if cat col fill leave for encoder(pd.get_dummies)
    num_cols= df.select_dtypes(include='number').columns
    num_cols=[col for col in num_cols if col!=target_col]
    df[num_cols]=df[num_cols].fillna(0)

    return df
        


