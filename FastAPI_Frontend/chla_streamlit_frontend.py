# chla_streamlit_frontend.py

import streamlit as st
import requests

def main():
    st.title("CHLA Patient No-Show Predictor")
    st.write("Enter appointment details to predict no-show probability:")

    # Input fields
    zipcode = st.text_input("ZIP Code", "90001")
    clinic = st.text_input("Clinic", "Pediatric Clinic")
    is_repeat = st.selectbox("Is Repeat Visit?", ["Yes", "No"])
    appt_type = st.text_input("Appointment Type", "General Consultation")
    ethnicity = st.text_input("Ethnicity", "Hispanic")
    race = st.text_input("Race", "White")
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    hour_of_day = st.number_input("Hour of Day (24hr format)", min_value=0, max_value=23, value=10)

    if st.button("Predict"):
        input_data = {
            "ZIPCODE": zipcode,
            "CLINIC": clinic,
            "IS_REPEAT": is_repeat,
            "APPT_TYPE_STANDARDIZE": appt_type,
            "ETHNICITY_STANDARDIZE": ethnicity,
            "RACE_STANDARDIZE": race,
            "AGE": age,
            "HOUR_OF_DAY": hour_of_day
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']} with {result['confidence']}% confidence.")
            else:
                st.error("Error from backend API. Please check if backend is running.")

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

if __name__ == "__main__":
    main()

