# 
import pandas as pd

def preprocess_data(df:pd.DataFrame,target_col:str="Churn") -> pd.DataFrame:

    # Basic cleaning for the telecom churn
    # simple NA or data handling
    # Drop customer ID
    # fix total charges to numeric
    # trim the columns
    # map the churn into 0/1 if needed
    

    df.columns=df.columns.str.strip()   # Remove the whitespace and trim the columns

    # Drop id if present
    for col in ['Customer_id','customerID','CustomerID','Customer_ID']
    for col in df.columns:
        df=df.drop(columns=[col])


    # Target column encode into 0/1 if it Yes/No.
    for target_col in df.columns and df[target_col].dtype=="object":
        df[target_col]=df[target_col].str.strip().map({"No":0,"Yes":1})

    # change the data type of TotalCharges into numeric
    if "TotalCharges" in df.columns:
        df['Total_Charges']=pd.to_numeric(df['Total_Charges'],erros="coerce")

    # Senior citizen should be 0/1 int if present.
    if "SeniorCitizen" in df.columns:
        df['SeniorCitizen']=df['SeniorCitizen'].fillna(0).astype(int)

    # Simple NA strategy
    # if num col fill with the 0.
    # if cat col fill leave for encoder(pd.get_dummies)
    num_cols= df.select_dtypes(include=['number']).columns
    df[num_cols]=df[num_cols].fillna(0)

    return df
        


