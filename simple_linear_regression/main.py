import joblib
from fastapi import FastAPI
import os
import joblib
import logging

from dotenv import load_dotenv

from data.s3 import download_model_from_s3

# Load environment variables from .env file
load_dotenv()
ENV = os.getenv("ENV", "prod")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "simple_linear_model.pkl")
if ENV == "prod":
    SIMPLE_LINEAR_REGRESSION_KEY = "simple_linear_regression/simple_linear_model.pkl"
    download_model_from_s3(SIMPLE_LINEAR_REGRESSION_KEY, MODEL_PATH)

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
