from pathlib import Path
import joblib

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose  import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,

)

from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.utils.validate_data import validate_telco_data

# Find project folder from the file's location
# train.py is inside the src/models  and it is parent's parent in the project root.
PROJECT_ROOT=Path(__file__).resolve().parents[2]

# location of the orignal raw file
DATA_PATH=PROJECT_ROOT/"data"/"raw"/"Customer_churn_data.csv"

# Save the model and preprocess together here.
MODEL_PATH=PROJECT_ROOT/"artifacts"/"churn_pipeline.joblib"



def train_model(df:pd.DataFrame,target_col:str="Churn") -> Pipeline:
    #  here we prepare the data ,train a churn model and return the trained model pipline.
    # 
    # Clean the data ,include convert the churn columns to binary 0/1.
    clean_df=preprocess_data(df,target_col=target_col)

    # Sepearte the ans column
    X=clean_df.drop(columns=[target_col])
    y=clean_df[target_col].asype(int)

    # Find which columns contain numeric and which colums categorical columns.
    numeric_cols=X.select_dtypes(include='number').columns.tolist()
    cat_cols=X.select_dtypes(include=["object","String","Category"]).columns.tolist()

    

