# ==============================
# File: deployment/fastapi_app.py
# ==============================

from fastapi import FastAPI
import pickle
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    with open('models/xgboost_model_best.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("XGBoost model loaded")
except Exception as e:
    xgb_model = None
    print(f"Failed to load XGBoost model: {e}")

try:
    tf_model = tf.keras.models.load_model('models/tensorflow_model.h5')
    tf_model.compile()  # ensure compile_metrics warning is suppressed
    print("TensorFlow model loaded")
except Exception as e:
    tf_model = None
    print(f"Failed to load TensorFlow model: {e}")

# Input schema
class InputData(BaseModel):
    features: list

@app.get("/")
async def root():
    return {"message": "Financial Risk Scoring Model API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict/xgboost")
async def predict_xgboost(input_data: InputData):
    if xgb_model is None:
        return {"error": "XGBoost model not loaded"}
    features = np.array(input_data.features).reshape(1, -1)
    prediction = xgb_model.predict(features)
    return {"xgboost_prediction": int(prediction[0])}

@app.post("/predict/tensorflow")
async def predict_tensorflow(input_data: InputData):
    if tf_model is None:
        return {"error": "TensorFlow model not loaded"}
    features = np.array(input_data.features).reshape(1, -1)
    prediction = tf_model.predict(features)
    return {"tensorflow_prediction": float(prediction[0][0])}

@app.get("/predict/sample")
async def predict_sample():
    if tf_model is None or xgb_model is None:
        return {"error": "One or more models not loaded"}
    X_test = np.load('data/processed/X_test.npy')
    idx = np.random.randint(0, X_test.shape[0])
    sample = X_test[idx].reshape(1, -1)
    xgb_pred = xgb_model.predict(sample)[0]
    tf_pred = float(tf_model.predict(sample)[0][0])
    return {
        "sample_input": X_test[idx].tolist(),
        "xgboost_prediction": int(xgb_pred),
        "tensorflow_prediction": tf_pred
    }

# Run with: uvicorn deployment.fastapi_app:app --reload --port 8000