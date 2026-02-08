from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import logging

app = FastAPI(title="Heart Disease Prediction")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully")

class Features(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/batch_pred")
def batch_predict(data: List[Features]):
    try:
        X = np.array([
            [
                r.age, r.sex, r.cp, r.trestbps, r.chol,
                r.fbs, r.restecg, r.thalach, r.exang,
                r.oldpeak, r.slope, r.ca, r.thal
            ]
            for r in data
        ])

        logger.info(f"Input shape: {X.shape}")
        preds = model.predict(X)

        return {"predictions": preds.tolist()}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
