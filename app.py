import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# -------------------------------
# Load secrets (admin credentials)
# -------------------------------
ADMIN_USERNAME = st.secrets["admin"]["username"]
ADMIN_PASSWORD = st.secrets["admin"]["password"]

# -------------------------------
# Inisialisasi state
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"  # default halaman awal

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# -------------------------------
# Fungsi untuk klasifikasi
# -------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open("model_menfess.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer_menfess.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
label_encoder = LabelEncoder()
label_encoder.classes_ = ['candaan', 'curhat', 'informasi', 'promosi', 'tanya']

def predict(texts):
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    return label_encoder.inverse_transform(y_pred)

# -------------------------------
# Sidebar Navigasi
# -------------------------------
st.sidebar.title("🔧 Menu")

if st.session_state.admin_logged_in:
    st.sidebar.success("✅ Admin mode aktif")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.admin_logged_in = False
        st.session_state.page = "home"
        st.rerun()
else:
    if st.sidebar.button("🔐 Login Admin"):
        st.session_state.page = "login"
        st.rerun()

if st.sidebar.button("🏠 Kembali ke Home"):
    st.session_state.page = "home"
    st.rerun()

# -------------------------------
# Halaman: Login Admin
# -------------------------------
if st.session_state.page == "login" and not st.session_state.admin_logged_in:
    st.title("🔐 Login Admin")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.success("Login berhasil!")
            st.session_state.admin_logged_in = True
            st.session_state.page = "admin"
            st.rerun()
        else:
            st.error("Username atau password salah.")

# -------------------------------
# Halaman: Admin (retrain)
# -------------------------------
elif st.session_state.page == "admin" and st.session_state.admin_logged_in:
    st.title("🔁 Retrain Model")
    uploaded_file = st.file_uploader("Upload dataset CSV (format: text,label)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
        new_vectorizer = TfidfVectorizer()
        X_train_vec = new_vectorizer.fit_transform(X_train)
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        new_model = MultinomialNB()
        new_model.fit(X_train_vec, y_train_enc)

        # Save model dan vectorizer
        with open("model.pkl", "wb") as f:
            pickle.dump(new_model, f)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(new_vectorizer, f)

        st.success("✅ Model berhasil diretrain dan disimpan.")

# -------------------------------
# Halaman: Home (klasifikasi)
# -------------------------------
else:
    st.title("📦 Menfess Classifier")
    menu = st.radio("Pilih input:", ["Teks Manual", "Upload CSV"])
    
    if menu == "Teks Manual":
        user_input = st.text_area("Masukkan menfess")
        if st.button("Klasifikasikan"):
            if user_input.strip() != "":
                result = predict([user_input])
                st.success(f"Prediksi kategori: **{result[0]}**")
    else:
        csv_file = st.file_uploader("Upload file CSV", type="csv")
        if csv_file:
            df = pd.read_csv(csv_file)
            if "text" in df.columns:
                preds = predict(df["text"])
                df["prediksi"] = preds
                st.write(df)
                st.download_button("⬇️ Download hasil", df.to_csv(index=False), "hasil_klasifikasi.csv")
            else:
                st.error("File harus memiliki kolom 'text'.")

