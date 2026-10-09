# Build Features

import pandas as pd

"""apply engineering features to the data frame. This is a placeholder for now, but can be expanded to include more complex feature engineering steps in the future.
"""

def _map_binary_series(s:pd.Series) ->pd.Series:
 """Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """

 # Gets unique remove values and NAN.
 vals=list(pd.Series(s.dropna().unique()).astype(str))
 valset=set(vals)

 # 
 if valset == {"Yes","No"}:
  return s.map({"No":0,"Yes":1}).astype("Int64")


 if len(vals)==2:
    
    
   # sort values to ensure consistent mapping across training and serving.
    sorted_vals=sorted(vals)
    mapping={sorted_vals[0]:0,sorted_vals[1]:1}
    return s.astype(str).map(mapping).astype("Int64")

 # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding or other methods later in the pipeline.
 return s

def build_features(df:pd.DataFrame,target_col:str="Churn") -> pd.DataFrame:
 """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """

 df=df.copy()
 print(f" starting feature enginnering on {df.shape[1]} columns...")

# Step 1- > identify Feature Types
# find categorical columns (object type) excluding the target variable 

 obj_cols=[c for c in df.select_dtypes(include=["object"]).columns if c!=target_col]
 numeric_cols=df.select_dtypes(include=["int64","float"]).columns.tolist()

 print(f" found  {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns")

 # Step 2--> split categorical by the cardinality.
 # Binary feature(exactly 2 unique value) get binary coding
 # multi-category feature (>2 feature values ) get one hot encoding

 binary_cols=[c for c in obj_cols if df[c].dropna().nunique()==2]
 multi_cols=[c for c in obj_cols if df[c].dropna().nunique()>2]

 print(f" Binary Features:{len(binary_cols)} | Multi-category features:{len(multi_cols)}")

 if binary_cols:
   print(f" Binary:{binary_cols}")
 if multi_cols:
  print(f" Multi_category:{multi_cols}")

  # Step 3- > apply binary coding
  # convert 2- category feature to 0/1 using deterministic mapping.
  for c in binary_cols:
    original_dtype=df[c].dtype
    df[c]=_map_binary_series(df[c].astype(str))
    print(f" {c}:{original_dtype} -> binary (0/1)")


# Step 4- Convert boolean columns
 bool_cols=df.select_dtypes(include=["bool"]).columns.tolist()
 if bool_cols:
   df[bool_cols]=df[bool_cols].astype(int)
   print(f" Converted {len(bool_cols)} boolean columns to int :{bool_cols}")

# Step 5- One hot encoding for multi_cols value....
# Critical :drop_first =True prevents multicollinarity.

 if multi_cols:
     
     print(f" Applying one-hot encoding to {len(multi_cols)} multi-category columns")
     original_shape=df.shape


     # Applying one hot encoding with drop_first=True for serving
     df=pd.get_dummies(df,columns=multi_cols,drop_first=True)

     new_features=df.shape[1]- original_shape[1] +len(multi_cols)
     print(f" created{new_features} new features from {len(multi_cols)} categorical columns")


     # Step 6 --> Data type Clean UP

 for c in binary_cols:
   if pd.api.types.is_integer_dtype(df[c]):
         # Fill any value with 0 and convert to int.
         df[c]=df[c].fillna(0).astype(int)

 print(f" Feature engineering complete: {df.shape[1]} final_features")
 return df

     
   
  









  

