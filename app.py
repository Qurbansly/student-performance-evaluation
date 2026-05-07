import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PISA Performance Prediction API")

model = joblib.load("models/pisa_xgboost_model.pkl")


class StudentRequest(BaseModel):
    ESCS: float
    HOMEPOS: float
    ICTRES: float
    ST004D01T: str
    ST005Q01JA: str
    ST006Q01JA: str
    ST006Q02JA: str
    ST006Q03JA: str
    ST007Q01JA: str


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: StudentRequest):
    student_data = pd.DataFrame([request.model_dump()])

    predicted_score = model.predict(student_data)[0]

    if predicted_score < 450:
        risk_level = "HIGH_RISK"
    elif predicted_score < 550:
        risk_level = "MEDIUM_RISK"
    else:
        risk_level = "LOW_RISK"

    return {
        "predicted_math_score": round(float(predicted_score), 2),
        "risk_level": risk_level
    }
