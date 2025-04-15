import pandas as pd
import numpy as np
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

with open(MODEL_PATH, "rb") as model_file:
    model = joblib.load(model_file)

with open(ENCODER_PATH, "rb") as encoder_file:
    encoder_dict = joblib.load(encoder_file)

# --- Categorical columns and model input order ---
category_col = ['ZIPCODE', 'CLINIC', 'IS_REPEAT', 'APPT_TYPE_STANDARDIZE', 
                'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']
expected_features = model.feature_names_in_

# --- Define the input schema ---
class AppointmentInput(BaseModel):
    ZIPCODE: str
    CLINIC: str
    IS_REPEAT: str
    APPT_TYPE_STANDARDIZE: str
    ETHNICITY_STANDARDIZE: str
    RACE_STANDARDIZE: str
    AGE: float
    HOUR_OF_DAY: int

# --- Preprocessing function ---
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Standardize text inputs for consistency
    for col in category_col:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.title()

    # Encode categorical features using pre-fitted encoders
    for col in category_col:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_ 
                else encoder_dict[col].transform(['Unknown'])[0])

    # Add missing columns with default value
    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    return df[expected_features]

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
