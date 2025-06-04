from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_model.pkl")
model_data = joblib.load(MODEL_PATH)

app = FastAPI()


# Define the expected input features
class HousingFeatures(BaseModel):
    MedInc: float
    AveRooms: float
    AveOccup: float


@app.get("/")
def root():
    return {"message": "Multivariable Linear Regression Model is live!"}


@app.post("/predict/")
def predict(features: HousingFeatures):
    x = np.array([features.MedInc, features.AveRooms, features.AveOccup])
    logging.info(f"Input: {x}")

    # Normalize
    x_norm = (x - model_data["mean"]) / model_data["std"]

    # Add bias (intercept) term
    x_with_bias = np.hstack([1.0, x_norm])

    # Predict
    y_pred = np.dot(x_with_bias, model_data["weights"])

    return {"predicted_house_value_100k": y_pred}


@app.get("/health")
def health():
    return {"status": "ok"}
