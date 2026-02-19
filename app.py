from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Loan Default Prediction API")

pipeline = joblib.load("loan_default_full_pipeline.pkl")

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
    return {"message": "Loan Default Prediction API running"}

@app.post("/predict")
def predict(data: LoanInput):
    arr = np.array([[*data.model_dump().values()]])
    prob = pipeline.predict_proba(arr)[:, 1][0]
    pred = int(prob >= 0.20)

    return {
        "default_probability": float(round(prob, 4)),
        "prediction": pred,
        "result": "Default Risk" if pred else "Safe Loan"
    }
