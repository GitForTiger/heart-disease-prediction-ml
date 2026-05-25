import streamlit as st
import pandas as pd
import joblib

# Load models and preprocessing objects
model = joblib.load("Heart_disease_KNN_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Provide the following details")

# Added missing text labels and commas
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholestrol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

# Separated the single string into three distinct list items
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)

# Fixed spelling of Angina
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
old_peak = st.slider("Old Peak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp, 
        'Cholesterol': cholestrol, # Check if dataset uses 'Cholesterol' or 'Cholestrol'
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': old_peak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain_type: 1,
        'RestingECG_' + resting_ecg: 1, 
        'ExerciseAngina_' + exercise_angina: 1, 
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns: 
        if col not in input_df.columns: 
            input_df[col] = 0
    
    input_df = input_df[expected_columns]

    # --- DEBUGGING SECTION ---
    st.write("### 🔍 Debugging Info")
    st.write("**1. What the Model Expects (From your Colab file):**", expected_columns)
    st.write("**2. What Streamlit is Sending (Look for mostly 0s):**", input_df)
    # -------------------------

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")