# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Title of the app
st.title("Prediksi Status Gizi Balita")

# Input fields for user
umur = st.number_input("Umur (bulan)", min_value=0)
jenis_kelamin = st.selectbox("Jenis Kelamin", options=["laki-laki", "perempuan"])
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0)

# Prepare input data
input_data = pd.DataFrame({
    'Umur (bulan)': [umur],
    'Jenis Kelamin': [1 if jenis_kelamin == 'perempuan' else 0],
    'Tinggi Badan (cm)': [tinggi_badan]
})

# Predict button
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    status_gizi = ["stunted", "normal", "tinggi", "severely stunted"]
    st.write(f"Status Gizi: {status_gizi[prediction[0]]}")