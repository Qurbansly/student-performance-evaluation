import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="PISA Student Performance Prediction API")

model = joblib.load("models/pisa_multi_output_model.pkl")


class StudentRequest(BaseModel):
    ESCS: float
    HOMEPOS: float
    ICTRES: float
    ICTHOME: float

    ST004D01T: str
    ST005Q01JA: str
    ST006Q01JA: str
    ST006Q02JA: str
    ST006Q03JA: str
    ST007Q01JA: str

    WORKPAY: str
    WORKHOME: str
    MATHPREF: str
    MATHEASE: str
    MATHMOT: str
    MATHPERS: str

    BELONG: float
    BULLIED: float
    FAMSUP: float
    TEACHSUP: float
    MATHEFF: float
    FAMCON: float
    ANXMAT: float


def classify_risk(score):
    if score < 400:
        return "HIGH_RISK"
    elif score < 500:
        return "MEDIUM_RISK"
    return "LOW_RISK"


@app.get("/")
def home():
    return {
        "message": "PISA 2022 Student Performance Prediction API is running"
    }


@app.post("/predict")
def predict(request: StudentRequest):
    input_df = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_df)[0]

    pv1 = float(prediction[0])
    pv2 = float(prediction[1])
    pv3 = float(prediction[2])
    pv4 = float(prediction[3])
    pv5 = float(prediction[4])

    average_score = float(prediction.mean())

    return {
        "PV1MATH": round(pv1, 2),
        "PV2MATH": round(pv2, 2),
        "PV3MATH": round(pv3, 2),
        "PV4MATH": round(pv4, 2),
        "PV5MATH": round(pv5, 2),
        "average_score": round(average_score, 2),
        "risk_level": classify_risk(average_score)
    }
