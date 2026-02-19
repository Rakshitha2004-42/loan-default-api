from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Prediction API")

# load trained pipeline
pipeline = joblib.load("loan_default_full_pipeline.pkl")


# ---------- INPUT SCHEMA ----------
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


# ---------- HOME ----------
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API running"}


# ---------- FEATURE ENGINEERING ----------
def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df["Loan_to_Income"] = df["LoanAmount"] / df["Income"]
    df["LoanTerm_to_Income"] = df["LoanTerm"] / df["Income"]
    df["EMI_Burden"] = df["LoanAmount"] / df["LoanTerm"]
    df["Debt_Stress"] = df["DTIRatio"] * df["InterestRate"]

    df["Income_per_CreditLine"] = df["Income"] / df["NumCreditLines"]
    df["Loan_per_CreditLine"] = df["LoanAmount"] / df["NumCreditLines"]

    df["Employment_Stability"] = df["MonthsEmployed"] / (df["Age"] + 1)
    df["LoanTerm_per_Age"] = df["LoanTerm"] / (df["Age"] + 1)

    # Credit score band
    df["CreditScore_Band"] = pd.cut(
        df["CreditScore"],
        bins=[0, 580, 670, 740, 800, 900],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df["LowCredit_HighDTI"] = ((df["CreditScore"] < 600) & (df["DTIRatio"] > 0.4)).astype(int)
    df["Unemployed_HighLoan"] = ((df["EmploymentType"] == "Unemployed") & (df["LoanAmount"] > 20000)).astype(int)
    df["HighInterest_LongTerm"] = ((df["InterestRate"] > 0.12) & (df["LoanTerm"] > 36)).astype(int)

    return df


# ---------- PREDICT ----------
@app.post("/predict")
def predict(data: LoanInput):

    # convert input → dataframe
    df = pd.DataFrame([data.model_dump()])

    # create engineered features
    df = create_features(df)

    # predict
    prob = pipeline.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.20)

    return {
        "default_probability": float(round(prob, 4)),
        "prediction": pred,
        "result": "Default Risk" if pred else "Safe Loan"
    }
