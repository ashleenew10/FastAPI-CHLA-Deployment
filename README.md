# FastAPI-CHLA-Deployment
# CHLA Patient No-Show Predictor

This project uses machine learning to predict whether a patient is likely to no-show for an upcoming appointment based on patient and appointment details.  
The app is deployed locally using a Streamlit frontend and a FastAPI backend, communicating through a JSON API.

---

## 📁 Project Structure

streamlit_handson/
├── FastAPI_Backend/
│   ├── prediction_backend_chla.py         # FastAPI backend script
│   ├── new_best_no_show_model.pkl         # Trained Random Forest model
│   ├── NEW_no_show_encoder.pkl            # Encoder dictionary for categorical features
│   └── requirements.txt                   # Dependencies for backend
│
├── FastAPI_Frontend/
│   └── chla_streamlit_frontend.py         # Streamlit frontend for user inputs
│
├── model_building/
│   └── CHLA_No_Show_Predictor_Ashlee.ipynb  # Notebook used to clean, train, and export model
│
└── README.md                              # This file

---

##  How to Run Locally

###  Prerequisites
- Python 3.12+
- pip
- All required packages listed in requirements.txt

---

###  Step-by-Step Instructions

1. Install dependencies:

   Open a terminal and run:
   pip install -r FastAPI_Backend/requirements.txt

   Includes: fastapi, uvicorn, joblib, scikit-learn, pandas, numpy, streamlit, etc.

2. Start the FastAPI Backend:

   In Terminal 1:
   cd FastAPI_Backend
   uvicorn prediction_backend_chla:app --reload

   Runs at: http://127.0.0.1:8000  
   Leave this terminal running.

3. Start the Streamlit Frontend:

   In Terminal 2 (new tab/window):
   cd FastAPI_Frontend
   streamlit run chla_streamlit_frontend.py

   App opens in browser at: http://localhost:8501  
   Fill in the form and click Predict to get no-show predictions.

---

## How It Works

- The user enters appointment data (e.g. age, clinic, appointment time, etc.) into the Streamlit form.
- Streamlit sends this data as a JSON request to the FastAPI backend.
- FastAPI loads the trained Random Forest model and label encoders.
- It makes a prediction and sends the result back to Streamlit.
- The user sees a prediction like "This patient will not show up" or "This patient will attend".

---

## Model Details

- Model: Random Forest Classifier
- Target: Whether the patient was a "No-Show" (1) or "Show-Up" (0)
- Features used: ZIP code, clinic, repeat visit, appointment type, ethnicity, race, age, appointment hour

Model was trained and tested in CHLA_No_Show_Predictor_Ashlee.ipynb.

---

## Troubleshooting

- If Streamlit frontend gives a connection error, make sure the backend is running on http://127.0.0.1:8000.
- If joblib.load() fails, ensure the .pkl model was trained and saved in the same Python/scikit-learn version you are using.
- You must keep both frontend and backend terminals running during local use.

---

## Built With

- Python
- FastAPI
- Streamlit
- Scikit-Learn
- Joblib
