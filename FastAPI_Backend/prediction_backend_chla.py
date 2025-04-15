# prediction_backend_chla.py

import pandas as pd
import numpy as np
import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# --- Setup FastAPI app ---
app = FastAPI()

# --- Load model and encoder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "new_best_no_show_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "NEW_no_show_encoder.pkl")

with open(MODEL_PATH, "rb") as file:
    model = joblib.load(file)

with open(ENCODER_PATH, "rb") as file:
    encoder_dict = joblib.load(file)

# --- Expected categorical features ---
category_col = ['ZIPCODE', 'CLINIC', 'IS_REPEAT', 'APPT_TYPE_STANDARDIZE', 'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']
expected_features = model.feature_names_in_

# --- Define the input format ---
class AppointmentInput(BaseModel):
    ZIPCODE: str
    CLINIC: str
    IS_REPEAT: str
    APPT_TYPE_STANDARDIZE: str
    ETHNICITY_STANDARDIZE: str
    RACE_STANDARDIZE: str
    AGE: float
    HOUR_OF_DAY: int

# --- Preprocess input ---
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Encode categorical variables
    for col in category_col:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_ 
                else encoder_dict[col].transform(['Unknown'])[0])

    # Fill missing columns
    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Reorder columns
    df = df[expected_features]

    return df

# --- Prediction endpoint ---
@app.post("/predict/")
async def predict_no_show(input_data: AppointmentInput):
    data_dict = input_data.dict()
    features = preprocess_input(data_dict)

    no_show_index = model.classes_.tolist().index(1)
    y_prob = model.predict_proba(features)[:, no_show_index][0]
    prediction = "No-Show" if y_prob >= 0.5 else "Show-Up"
    confidence = round((y_prob if prediction == "No-Show" else 1 - y_prob) * 100, 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }
