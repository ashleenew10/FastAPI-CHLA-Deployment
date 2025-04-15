import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import os

# --- Define relative paths safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Always find path based on current .py location
MODEL_PATH = os.path.join(BASE_DIR, "best_no_show_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "NEW_no_show_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "NEW_CLEAN_CHLA_clean_data_2024_Appointments.csv")

# --- Load the trained model ---
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)  

# --- Load the encoder dictionary ---
with open(ENCODER_PATH, "rb") as encoder_file:
    encoder_dict = pickle.load(encoder_file)

# --- Load the cleaned 2024 dataset ---
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df_2024 = load_data()

# --- Expected categorical features that need encoding ---
category_col = ['ZIPCODE', 'CLINIC', 'IS_REPEAT', 'APPT_TYPE_STANDARDIZE', 
                'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']

# --- Expected features in the trained model ---
expected_features = model.feature_names_in_

# --- Function to preprocess input data ---
def preprocess_input(df, encoder_dict):
    df = df.copy()

    # Encode categorical variables using stored encoders
    for col in category_col:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_ 
                else encoder_dict[col].transform(['Unknown'])[0])

    # Ensure dataset has all features the model expects
    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Fill missing columns with default value 0

    # Reorder columns to match model training
    df = df[expected_features]
    
    return df

# --- Function to make predictions ---
def predict_no_show(features):
    no_show_index = model.classes_.tolist().index(1)  # Get index of "No-Show" class
    y_prob = model.predict_proba(features)[:, no_show_index]  # Probability of No-Show
    y_pred = np.where(y_prob >= 0.5, "No-Show", "Show-Up")
    return y_pred, y_prob

# --- Streamlit App ---
def main():
    st.title("CHLA Patient No-Show Prediction")
    st.write("Select clinic and appointment date range to view no-show predictions.")

    # User inputs
    clinic_name = st.selectbox("Select Clinic", df_2024["CLINIC"].unique())

    # Get available date range
    min_date = pd.to_datetime(df_2024["BOOK_DATE"]).min()
    max_date = pd.to_datetime(df_2024["APPT_DATE"]).max()

    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=max_date)

    # Predict Button
    if st.button("Get Predictions"):
        df_filtered = df_2024[(df_2024["CLINIC"] == clinic_name) & 
                              (pd.to_datetime(df_2024["APPT_DATE"]) >= pd.to_datetime(start_date)) & 
                              (pd.to_datetime(df_2024["APPT_DATE"]) <= pd.to_datetime(end_date))]

        if df_filtered.empty:
            st.warning("No appointments found for the selected clinic and date range.")
        else:
            # Keep necessary identifiers for output
            output_data = df_filtered[["MRN", "APPT_ID", "APPT_DATE", "HOUR_OF_DAY"]].copy()

            X_input = preprocess_input(df_filtered, encoder_dict)

            # Get predictions
            y_pred, y_prob = predict_no_show(X_input)

            # Add predictions to output
            output_data["Prediction"] = y_pred
            output_data["MRN"] = output_data["MRN"].astype(str)
            output_data["APPT_ID"] = output_data["APPT_ID"].astype(str)

            # Confidence score
            output_data["Confidence"] = [
                (p if pred == 'No-Show' else 1 - p) * 100
                for pred, p in zip(y_pred, y_prob)]

            # Display results
            st.subheader("Predicted No-Show Appointments")
            st.dataframe(output_data[["MRN", "APPT_ID", "APPT_DATE", "HOUR_OF_DAY", "Prediction", "Confidence"]])

if __name__ == "__main__":
    main()

