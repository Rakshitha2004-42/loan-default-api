from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Loan Default Prediction API")

# ---- Fix: correct model path for Render ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_default_full_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)
# -------------------------------------------


class LoanInput(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: int
    EmploymentType: int
    MaritalStatus: int
    HasMortgage: int
    HasDependents: int
    LoanPurpose: int
    HasCoSigner: int


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(data: LoanInput):
    df = pd.DataFrame([data.model_dump()])
    prob = pipeline.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.20)

    return {
        "default_probability": float(round(prob, 4)),
        "prediction": pred,
        "result": "Default Risk" if pred == 1 else "Safe Loan"
    }
