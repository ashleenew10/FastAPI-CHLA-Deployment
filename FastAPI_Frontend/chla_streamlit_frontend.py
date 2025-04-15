import streamlit as st
import joblib  # <-- switched from pickle to joblib
import numpy as np
import pandas as pd
import os

# --- Define file paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model_building", "new_best_no_show_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "model_building", "NEW_no_show_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "model_building", "NEW_CLEAN_CHLA_clean_data_2024_Appointments.csv")

# --- Load model, encoder, and data ---
with open(MODEL_PATH, "rb") as file:
    model = joblib.load(file)

with open(ENCODER_PATH, "rb") as encoder_file:
    encoder_dict = joblib.load(encoder_file)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df_2024 = load_data()

# --- Configurations ---
category_col = ['ZIPCODE', 'CLINIC', 'IS_REPEAT', 'APPT_TYPE_STANDARDIZE',
                'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']

expected_features = model.feature_names_in_

# --- Preprocessing Function ---
def preprocess_input(df, encoder_dict):
    df = df.copy()

    for col in category_col:
        if col in df.columns:
            df[col] = df[col].apply(lambda x:
                encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_
                else encoder_dict[col].transform(['Unknown'])[0])

    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    return df[expected_features]

# --- Prediction Function ---
def predict_no_show(features):
    no_show_index = model.classes_.tolist().index(1)
    y_prob_no_show = model.predict_proba(features)[:, no_show_index]
    y_prob_show = 1 - y_prob_no_show  # Invert
    y_pred = np.where(y_prob_no_show >= 0.5, "No-Show", "Show-Up")
    return y_pred, np.round(y_prob_show * 100, 2)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="CHLA No-Show Predictor", layout="centered")
    st.title("CHLA Patient No-Show Prediction")
    st.write("Select a clinic and date range to view predicted no-show appointments.")

    clinic_name = st.selectbox("Select Clinic", sorted(df_2024["CLINIC"].dropna().unique()))

    min_date = pd.to_datetime(df_2024["BOOK_DATE"]).min()
    max_date = pd.to_datetime(df_2024["APPT_DATE"]).max()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=max_date)

    if st.button("Get Predictions"):
        df_filtered = df_2024[
            (df_2024["CLINIC"] == clinic_name) &
            (pd.to_datetime(df_2024["APPT_DATE"]) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(df_2024["APPT_DATE"]) <= pd.to_datetime(end_date))
        ]

        if df_filtered.empty:
            st.warning("No appointments found for the selected clinic and date range.")
        else:
            output_data = df_filtered[["MRN", "APPT_ID", "APPT_DATE", "HOUR_OF_DAY"]].copy()
            X_input = preprocess_input(df_filtered, encoder_dict)
            y_pred, y_prob = predict_no_show(X_input)

            output_data["No-Show Prediction"] = y_pred
            output_data["Probability of Show-Up (%)"] = y_prob

            st.subheader("Predicted No-Show Appointments")
            st.dataframe(output_data)

if __name__ == "__main__":
    main()

