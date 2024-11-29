import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('trained_model.pkl')

# Title of the app
st.title("Prediksi Status Gizi Balita")

# Navigation menu
menu = ["Deskripsi", "Sumber Dataset", "Tampilan Dataset", "Visualisasi Data", "Hasil Prediksi"]
choice = st.sidebar.selectbox("Navigasi", menu)

if choice == "Deskripsi":
    st.subheader("Deskripsi")
    st.write("Aplikasi ini digunakan untuk memprediksi status gizi balita berdasarkan umur, jenis kelamin, dan tinggi badan. Model yang digunakan telah dilatih menggunakan dataset terpercaya.")

elif choice == "Sumber Dataset":
    st.subheader("Sumber Dataset")
    st.write("Dataset diperoleh dari [example.com/dataset](https://example.com/dataset). Dataset ini berisi data status gizi balita yang mencakup berbagai atribut seperti umur, jenis kelamin, dan tinggi badan.")

elif choice == "Tampilan Dataset":
    st.subheader("Tampilan Dataset")
    example_data = pd.DataFrame({
        'Umur (bulan)': [6, 12, 18, 24],
        'Jenis Kelamin': [0, 1, 0, 1],
        'Tinggi Badan (cm)': [60, 75, 80, 85],
        'Status Gizi': ["normal", "tinggi", "stunted", "severely stunted"]
    })
    st.dataframe(example_data)

elif choice == "Visualisasi Data":
    st.subheader("Visualisasi Data")
    example_data = pd.DataFrame({
        'Umur (bulan)': [6, 12, 18, 24],
        'Tinggi Badan (cm)': [60, 75, 80, 85]
    })
    fig, ax = plt.subplots()
    ax.bar(example_data['Umur (bulan)'], example_data['Tinggi Badan (cm)'], color='skyblue')
    ax.set_title("Hubungan Umur dan Tinggi Badan")
    ax.set_xlabel("Umur (bulan)")
    ax.set_ylabel("Tinggi Badan (cm)")
    st.pyplot(fig)

elif choice == "Hasil Prediksi":
    st.subheader("Hasil Prediksi")
    umur = st.number_input("Umur (bulan)", min_value=0, value=0)
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=["laki-laki", "perempuan"], index=0)
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0, value=0)

    # Prepare input data
    input_data = pd.DataFrame({
        'Umur (bulan)': [umur],
        'Jenis Kelamin': [1 if jenis_kelamin == 'perempuan' else 0],
        'Tinggi Badan (cm)': [tinggi_badan]
    })

    if st.button("Prediksi"):
        prediction = model.predict(input_data)
        status_gizi = ["stunted", "normal", "tinggi", "severely stunted"]
        st.write(f"Status Gizi: {status_gizi[prediction[0]]}")
