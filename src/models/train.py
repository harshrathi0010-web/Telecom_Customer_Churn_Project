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
    y=clean_df[target_col].astype(int)

    # Find which columns contain numeric and which colums categorical columns.
    numeric_cols=X.select_dtypes(include='number').columns.tolist()
    categorical_cols=X.select_dtypes(include=["object","string","category"]).columns.tolist()

    # For numeric values fill with the median value and then scale the data.
    numeric_pipeline=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )


    # For categorical values text values first  fill values then convert into the numbere.
    categorical_pipeline=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ]
    ) 

    # Apply the appropriate preprocessing steps to the each type of the column....
    preprocessing=ColumnTransformer(
        transformers=[
        ("numeric",numeric_pipeline,numeric_cols),
        ("category",categorical_pipeline,categorical_cols,),
        ]
    )

    # The piepline first preprocess the inputs then the logistic regression 
    # class-weight ="balanced"helps account for the fewer churner than the non churner

    model=Pipeline(
        steps=[
            ("preprocessor",preprocessing),
            ("classifier", LogisticRegression(class_weight="balanced",max_iter=1000,random_state=23,
                ),
             ),
        ]
    )


    # keep some rows for the evaluation
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22,stratify=y)

    # Leanr from the training rows
    model.fit(X_train,y_train)

    # Predict churn labels and churn prob for the held-back rows.
    predictions=model.predict(X_test)
    churn_probabilites=model.predict_proba(X_test)[:,1]

    # Print metrics that summarize the model performance.
    print(f"accuracy:{accuracy_score(y_test,predictions):.3f}")
    print(f"Precision:{precision_score(y_test,predictions,zero_division=0):.3f}")
    print(f"Recall:{recall_score(y_test,predictions,zero_division=0):.3f}")
    print(f"F1 score:{f1_score(y_test,predictions,zero_division=0):.3f}")
    print(f"ROC-AUC:{roc_auc_score(y_test,churn_probabilites):.3f}")


    # Show correct and inccorect predicition for each class
    print("\nConfusion_Matrix")
    print(confusion_matrix(y_test,predictions))

    # Show precision ,f1, recall for each.
    print("\nClassification_report")
    print(classification_report(y_test,predictions,zero_division=0))


    report_text = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

    report_path = MODEL_PATH.parent / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Report saved to: {report_path}")

    # Create the artifacts folder if it does not exist.
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)

    # Save the model pipeline ,include preprocessing and the trained model.
    joblib.dump(model,MODEL_PATH)
    print(f"\n saved model to :{MODEL_PATH}")

    return model

def main() -> None:
    # Running the training process from the project raw CSV file.

    # Read the CSV into a DataFrame
    raw_df=load_data(str(DATA_PATH))

    # check the original data before cleaning.
    is_valid,errors=validate_telco_data(raw_df)

    # stop training if the data did not pass validation
    if not  is_valid:
        raise ValueError(f"Data validation failed:{errors}")

    # train,evaluate and save model
    train_model(raw_df)


# Run model only when the file is starting as a module scripts
if __name__=="__main__":
    main()






    

