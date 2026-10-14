# 
# Telecom Customer Churn Prediction

This project predicts whether a telecom customer may churn using a **Logistic Regression** model. It includes a data and training pipeline, a FastAPI service, and a simple web form.

## Business problem and solution

Customer churn reduces recurring revenue. This project estimates churn risk from customer account and service details so the retention team can prioritize follow-up. Predictions support staff decisions; they do not prove why a customer may leave.

## Run locally

Requirements: Python 3.12 and the authorized dataset at `data/raw/Customer_churn_data.csv` (the data is not included in Git).

From the project root in PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
python -m uvicorn src.app.main:app --reload
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the form or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for API documentation. Training saves `artifacts/churn_pipeline.joblib` and `artifacts/classification_report.txt`.

## Docker and AWS EC2

Build and run the Docker image locally:

```powershell
docker build -t telecom-churn:local .
docker run --rm -p 8000:8000 telecom-churn:local
```

The image was pushed to Docker Hub as `harshrathi10/telecom-customer-churn-project:latest`. To deploy it on an AWS EC2 Linux instance:

1. Install and start Docker on the instance.
2. Allow inbound access to port `8000` in the EC2 security group for your intended users.
3. Run:

   ```bash
   docker pull harshrathi10/telecom-customer-churn-project:latest
   docker run -d --restart unless-stopped -p 8000:8000 harshrathi10/telecom-customer-churn-project:latest
   ```

Open `http://<EC2-PUBLIC-IP>:8000/`. Add the real URL here after the EC2 deployment is complete. Automated CI/CD is future work.

## Stakeholder recommendations

- Use the churn score to prioritize human follow-up, not to make automatic customer decisions.
- Pilot retention offers and measure saved customers, customer experience, and campaign cost.
- The current holdout report shows **81% recall** and **54% precision** for churn. This catches many churners but also flags some customers who would stay; choose an operating threshold based on outreach capacity and business costs.
- Monitor performance on recent customers and investigate customer context with domain teams. The model estimates risk; it does not establish causes.
