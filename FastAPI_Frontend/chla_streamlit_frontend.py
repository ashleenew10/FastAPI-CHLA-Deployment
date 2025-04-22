import streamlit as st
st.set_page_config(page_title="CHLA No-Show Predictor (API)", layout="centered")  

import pandas as pd
import requests
import os

# --- Define file paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "NEW_CLEAN_CHLA_clean_data_2024_Appointments.csv")

# --- Load the cleaned 2024 dataset ---
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df_2024 = load_data()

# --- Streamlit App ---
def main():
    st.title("CHLA Patient No-Show Prediction (via FastAPI)")
    st.write("Select a clinic and date range to view predicted no-show appointments using the backend API.")

    # Clinic selection
    clinic_name = st.selectbox("Select Clinic", sorted(df_2024["CLINIC"].dropna().unique()))

    # Date range inputs
    min_date = pd.to_datetime(df_2024["BOOK_DATE"]).min()
    max_date = pd.to_datetime(df_2024["APPT_DATE"]).max()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=max_date)

    if st.button("Get Predictions"):
        # Filter rows by selected clinic and date range
        df_filtered = df_2024[
            (df_2024["CLINIC"] == clinic_name) &
            (pd.to_datetime(df_2024["APPT_DATE"]) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(df_2024["APPT_DATE"]) <= pd.to_datetime(end_date))
        ]

        if df_filtered.empty:
            st.warning("No appointments found for the selected clinic and date range.")
        else:
            output_data = df_filtered[["MRN", "APPT_ID", "APPT_DATE", "HOUR_OF_DAY"]].copy()
            predictions = []
            confidences = []

            # Loop through each row and send request to backend
            for _, row in df_filtered.iterrows():
                payload = {
                    "ZIPCODE": str(row["ZIPCODE"]),
                    "CLINIC": row["CLINIC"],
                    "IS_REPEAT": row["IS_REPEAT"],
                    "APPT_TYPE_STANDARDIZE": row["APPT_TYPE_STANDARDIZE"],
                    "ETHNICITY_STANDARDIZE": row["ETHNICITY_STANDARDIZE"],
                    "RACE_STANDARDIZE": row["RACE_STANDARDIZE"],
                    "AGE": row["AGE"],
                    "HOUR_OF_DAY": row["HOUR_OF_DAY"]
                }

                try:
                    # Use "backend" if using Docker Compose network
                    response = requests.post("http://localhost:8000/predict/", json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        predictions.append(result["prediction"])
                        confidences.append(result["confidence"])
                    else:
                        predictions.append("Error")
                        confidences.append(None)
                except Exception as e:
                    predictions.append("Error")
                    confidences.append(None)

            # Add results to output
            output_data["No-Show Prediction"] = predictions
            output_data["Probability of Show-Up (%)"] = confidences

            st.subheader("Predicted No-Show Appointments")
            st.dataframe(output_data)

if __name__ == "__main__":
    main()


