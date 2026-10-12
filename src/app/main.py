from typing import Literal
from fastapi import FastAPI,HTTPException

from pydantic import BaseModel,Field

from src.serving.predict import predict_customer

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR=Path(__file__).parent / "static"
app=FastAPI(
    

    title="Telecom  Customer Churn Prediction",
    description="Predict whether the customer wil churn or not",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR),name="static")

# FASTAPI (pydantic) check whether customer details have valid types and value.

class CustomerData(BaseModel):

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes","No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get('/',include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get('/health')
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post('/predict')
def predict(data: CustomerData) -> dict:
    try:
        customer_details = (
            data.model_dump() if hasattr(data, "model_dump") else data.dict()
        )
        return predict_customer(customer_details)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

        





    