import streamlit as st
import joblib

# Load model dan tools
model = joblib.load("model_menfess.pkl")
vectorizer = joblib.load("vectorizer_menfess.pkl")
encoder = joblib.load("label_encoder_menfess.pkl")

# UI
st.set_page_config(page_title="Klasifikasi Menfess", page_icon="🤖")
st.title("📬 Klasifikasi Menfess Otomatis")
st.markdown("Masukkan teks menfess di bawah ini, dan sistem akan mengkategorikannya.")

text_input = st.text_area("Tulis menfess kamu di sini:", height=150)

if st.button("Klasifikasikan"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        vectorized = vectorizer.transform([text_input])
        prediction = model.predict(vectorized)
        label = encoder.inverse_transform(prediction)[0]

        st.success(f"Kategori menfess: **{label}**")
