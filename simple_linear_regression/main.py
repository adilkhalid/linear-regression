import joblib
from fastapi import FastAPI
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "simple_linear_model.pkl")
model_data = joblib.load(MODEL_PATH)
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Model is live!"}


@app.post("/predict/")
def predict(medinc: float):
    logging.info(f"Medium Income: {medinc}")
    x = (medinc - model_data["mean"]) / model_data["std"]
    y_pred = model_data["slope"] * x + model_data["intercept"]
    return {"predicted_house_value_100k": y_pred}


@app.get("/health")
def health():
    return {"status": "ok"}
