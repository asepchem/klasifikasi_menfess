# 📦 Menfess Classifier Web App

![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-yellow?logo=scikit-learn)

Aplikasi web untuk mengklasifikasi "menfess" (mention confession) ke dalam beberapa kategori seperti `curhat`, `candaan`, `promosi`, `tanya`, dan `informasi`, menggunakan machine learning. Dibangun dengan Python, scikit-learn, dan Streamlit.

---

## 🧠 Tahap 1: Data Preparation

### 1.1 Desain Label Kategori
- `curhat`: Cerita personal, perasaan, unek-unek.
- `candaan`: Humor, jokes, sarkas.
- `tanya`: Pertanyaan/opini.
- `promosi`: Promosi akun, jasa, event.
- `informasi`: Berita, info, dan update.

### 1.2 Format Dataset
CSV dengan dua kolom:
```csv
text,label
"Isi menfess di sini","curhat"
```

### 1.3 Generate Data Dummy
- Menggunakan `random.choices()` dan template string per label.
- 200 contoh data per label.


## 🤖 Tahap 2: Model Training

### 2.1 Preprocessing
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
```
- TF-IDF untuk konversi teks ke vektor numerik.
- LabelEncoder untuk mengubah label string menjadi angka.

### 2.2 Train-Test Split
```python
from sklearn.model_selection import train_test_split
```

### 2.3 Model Training
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 2.4 Evaluasi Akurasi
```python
from sklearn.metrics import accuracy_score, classification_report
```

---

## 🌐 Tahap 3: Web App dengan Streamlit

### 3.1 Struktur File
```
📂 menfess-classifier/
├── model.pkl
├── vectorizer.pkl
├── app.py
├── dataset.csv
```

### 3.2 Streamlit App (app.py)
- Input teks manual
- Upload file CSV untuk klasifikasi massal
- Tampilkan hasil prediksi

```bash
streamlit run app.py
```

### 3.3 Deploy ke Streamlit Cloud
- Upload ke GitHub
- Login ke [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Pilih repo dan deploy 🎉

---

## 💾 Menyimpan Model
```python
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
```

---

## 🔍 Testing & Evaluasi
- Evaluasi manual dengan input real dari pengguna
- Uji klasifikasi massal dengan upload CSV
- Cek metrik klasifikasi dan confusion matrix

---

## ✨ Pengembangan Lanjutan (Opsional)
- Visualisasi hasil klasifikasi
- Tambahkan fitur retraining dari web
- Gunakan FastAPI untuk backend
- Export hasil klasifikasi ke Excel/CSV

---

## 👨‍💻 Author
Project pribadi oleh Asep Saefuddin Ash Shidiq

---

## 🧑‍💻 Cara Kontribusi
1. Fork repo ini
2. Buat branch baru `fitur-xyz`
3. Commit dan buat pull request
4. Sertakan deskripsi perubahan

---
