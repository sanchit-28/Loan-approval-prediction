"""
Score a single loan applicant with the trained model (run src/loan_approval.py first).

Usage:
    python src/predict.py                       # scores two built-in demo applicants
    from predict import predict_applicant       # use it from your own code
"""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "loan_approval_model.joblib"


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Same banking ratios as in loan_approval.py (keep the two in sync)."""
    d = d.copy()
    d["Total_Income"] = d.Applicant_Income + d.Coapplicant_Income
    d["Loan_to_Income"] = d.Loan_Amount / d.Total_Income
    d["Approx_EMI"] = d.Loan_Amount / d.Loan_Term
    d["EMI_to_Income"] = d.Approx_EMI / d.Total_Income
    d["Collateral_Coverage"] = d.Collateral_Value / d.Loan_Amount
    d["Savings_to_Loan"] = d.Savings / d.Loan_Amount
    d["Income_per_Dependent"] = d.Total_Income / (d.Dependents + 1)
    return d


def predict_applicant(applicant: dict) -> dict:
    artefact = joblib.load(MODEL_PATH)
    p = artefact["model"].predict_proba(add_features(pd.DataFrame([applicant])))[0, 1]
    return {"approval_probability": round(float(p), 3),
            "decision": "APPROVE" if p >= artefact["threshold"] else "REJECT"}


if __name__ == "__main__":
    strong = dict(Applicant_Income=9500, Coapplicant_Income=3000, Employment_Status="Salaried", Age=34,
                  Marital_Status="Married", Dependents=1, Credit_Score=720, Existing_Loans=1, DTI_Ratio=0.25,
                  Savings=12000, Collateral_Value=30000, Loan_Amount=18000, Loan_Term=48, Loan_Purpose="Home",
                  Property_Area="Urban", Education_Level="Graduate", Gender="Female", Employer_Category="Private")
    weak = dict(strong, Credit_Score=610, DTI_Ratio=0.45)
    print("Strong applicant ->", predict_applicant(strong))
    print("Weak applicant   ->", predict_applicant(weak))
