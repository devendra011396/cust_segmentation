import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessor
model = joblib.load("model/customer_segment_model.joblib")
preprocessor = joblib.load("model/scaler.joblib")

st.title("🧠 Customer Segmentation Predictor")

st.markdown("Fill in the customer details below to see which segment they belong to:")

# Input features
age = st.slider("Age", 18, 100, 30)
income = st.slider("Annual Income (₹)", 10000, 200000, 50000, step=1000)
score = st.slider("Spending Score", 0, 100, 50)
freq = st.slider("Purchase Frequency (monthly)", 1, 30, 10)
spend = st.slider("Total Spending (₹)", 0.0, 50000.0, 10000.0)
family = st.slider("Family Size", 1, 10, 3)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Widowed", "Divorced"])

# Predict button
if st.button("Predict Segment"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Annual_Income": income,
        "Spending_Score": score,
        "Purchase_Frequency": freq,
        "Total_Spending": spend,
        "Family_Size": family,
        "Gender": gender,
        "Marital_Status": marital
    }])

    # Preprocess and predict
    transformed = preprocessor.transform(input_df)
    cluster = model.predict(transformed)[0]

    st.success(f"The customer belongs to **Segment {cluster}**.")
