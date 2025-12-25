# Emotion Analysis App

Aplikasi analisis emosi dari teks menggunakan Machine Learning dan Streamlit.

## Persyaratan

- Python 3.8 atau lebih tinggi
- pip (Python package installer)

## Instalasi dan Menjalankan Aplikasi

### 1. Clone atau Download Project

git clone <repository-url>
cd emotion-analysis-app


### 2. Install Dependencies
pip install -r requirements.txt


### 3. Menyiapkan Model dan Data
Pastikan folder `models/` berisi file-file berikut:
- `best_emotion_model.pkl` (model terbaik)
- `text_preprocessor.pkl` (text preprocessor)
- `tfidf_vectorizer.pkl` (TF-IDF vectorizer)
- `label_mapping.pkl` (label mapping)
- `model_metadata.json` (metadata model)

Pastikan folder `data/` berisi:
- `cleaned_emotions.csv` (dataset)

### 4. Menjalankan Aplikasi
streamlit run app.py


Aplikasi akan berjalan di `http://localhost:8501`

## 🎯 Fitur Aplikasi

### 1. 🔮 Single Prediction
- Analisis emosi dari teks tunggal
- Input manual atau upload file
- Visualisasi probabilitas
- WordCloud dari teks

### 2. 📈 Batch Analysis
- Analisis emosi dari file CSV
- Download hasil sebagai CSV
- Visualisasi distribusi emosi

### 3. 📊 Model Info
- Informasi performa model
- Statistik dataset
- Detail preprocessing

### 4. 📁 About Dataset
- Preview dataset
- Distribusi emosi
- Statistik dataset

## 🧠 Model Machine Learning

- **Algoritma:** Random Forest (Tuned)
- **Metrik Evaluasi:**
  - Accuracy: [nilai dari metadata]
  - F1-Score: [nilai dari metadata]
  - Precision: [nilai dari metadata]
  - Recall: [nilai dari metadata]
- **Preprocessing:** Text cleaning, stopword removal, lemmatization
- **Vectorization:** TF-IDF dengan 5000 fitur

## 📊 Dataset

- **Sumber:** Dataset emosi teks
- **Jumlah Sample:** [jumlah dari metadata]
- **Jumlah Kelas:** 6 (sadness, joy, love, anger, fear, surprise)
- **Fitur:** Teks mentah

## 👨‍💻 Developer

**Nama:** Muhammad Galid Avero  
**NIM:** 2311532008  
**Mata Kuliah:** Praktikum Big Data
