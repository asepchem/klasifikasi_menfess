import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load secrets (from .streamlit/secrets.toml)
ADMIN_USERNAME = st.secrets["admin"]["username"]
ADMIN_PASSWORD = st.secrets["admin"]["password"]

# Session state untuk login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login berhasil!")
        else:
            st.error("Username atau password salah.")

def logout():
    st.session_state.logged_in = False
    st.success("Berhasil logout!")

# Load model & vectorizer
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def save_model(model, vectorizer):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

# Halaman utama setelah login
def main_app():
    st.title("📦 Menfess Classifier Web App")

    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih fitur:", ["Klasifikasi Teks", "Klasifikasi Massal", "Retrain Model", "Logout"])

    model, vectorizer = load_model()

    if page == "Klasifikasi Teks":
        st.header("📝 Input Teks")
        input_text = st.text_area("Masukkan teks menfess:")
        if st.button("Klasifikasi"):
            if input_text:
                X = vectorizer.transform([input_text])
                prediction = model.predict(X)[0]
                st.success(f"Prediksi kategori: `{prediction}`")

    elif page == "Klasifikasi Massal":
        st.header("📁 Upload CSV")
        file = st.file_uploader("Upload file CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            if "text" in df.columns:
                X = vectorizer.transform(df["text"])
                preds = model.predict(X)
                df["prediksi"] = preds
                st.dataframe(df)
                st.download_button("📥 Download Hasil", df.to_csv(index=False), "hasil_klasifikasi.csv")
            else:
                st.error("CSV harus memiliki kolom 'text'")

    elif page == "Retrain Model":
        st.header("♻️ Upload Data Baru untuk Training")
        file = st.file_uploader("Upload CSV berisi `text` dan `label`", type=["csv"], key="retrain")
        if file:
            df = pd.read_csv(file)
            if "text" in df.columns and "label" in df.columns:
                st.write("Contoh data:")
                st.dataframe(df.head())

                if st.button("🚀 Mulai Training Ulang"):
                    st.info("Melatih model baru...")
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(df["text"])
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(df["label"])
                    model = MultinomialNB()
                    model.fit(X, y)
                    save_model(model, vectorizer)
                    st.success("Model berhasil dilatih ulang dan disimpan!")
            else:
                st.error("CSV harus memiliki kolom 'text' dan 'label'.")

    elif page == "Logout":
        logout()

# Mulai dari sini
if st.session_state.logged_in:
    main_app()
else:
    login()
