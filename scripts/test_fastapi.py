# Lets the scripts send a request to the FASTAPI server to check.
import requests


# Address of the FASTAPI predict endpoint.
API_URL = "http://127.0.0.1:8000/predict"

# Sample customer fields expected by CustomerData in main.py
sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75,
}

# This function send a POST request to the FAST API and prins the response
def main():
    try:
        response = requests.post(API_URL, json=sample_data, timeout=10)
    except requests.exceptions.RequestException as error:
        print(f"Could not reach the API: {error}")
        print("Make sure Uvicorn is running, then try again.")
        return

    print("Status code:", response.status_code)
    print("Response:", response.text)

    if response.status_code == 200:
        print("API check succeeded.")
    else:
        print("API returned an error. Check the response above.")


if __name__ == "__main__":
    main()