from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Prediction API")

# Load trained pipeline
pipeline = joblib.load("loan_default_full_pipeline.pkl")


# ----------- INPUT SCHEMA (RAW USER INPUT) -----------
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
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# ----------- CATEGORY MAPPINGS -----------
education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}

employment_map = {
    "Unemployed": 0,
    "Part-time": 1,
    "Full-time": 2,
    "Self-employed": 3
}

marital_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}

purpose_map = {
    "Personal": 0,
    "Home": 1,
    "Auto": 2,
    "Education": 3,
    "Business": 4
}


# ----------- HOME ROUTE -----------
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API running"}


# ----------- PREDICTION ROUTE -----------
@app.post("/predict")
def predict(data: LoanInput):

    # Convert categorical text → numeric
    processed_data = {
        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
        "InterestRate": data.InterestRate,
        "LoanTerm": data.LoanTerm,
        "DTIRatio": data.DTIRatio,
        "Education": education_map.get(data.Education, 0),
        "EmploymentType": employment_map.get(data.EmploymentType, 0),
        "MaritalStatus": marital_map.get(data.MaritalStatus, 0),
        "HasMortgage": yes_no_map.get(data.HasMortgage, 0),
        "HasDependents": yes_no_map.get(data.HasDependents, 0),
        "LoanPurpose": purpose_map.get(data.LoanPurpose, 0),
        "HasCoSigner": yes_no_map.get(data.HasCoSigner, 0),
    }

    # Convert to DataFrame (required for sklearn pipeline)
    df = pd.DataFrame([processed_data])

    # Prediction
    prob = pipeline.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.20)

    return {
        "default_probability": round(float(prob), 4),
        "prediction": pred,
        "result": "Default Risk" if pred else "Safe Loan"
    }
