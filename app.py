import streamlit as st
import cv2
import numpy as np
import pyfeats
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Sistem Deteksi Penyakit Paru-Paru", layout="wide")

st.title("🩺 Sistem Deteksi Penyakit Paru-Paru pada Citra X-Ray Dada")
st.markdown("""
Aplikasi ini menggunakan gabungan fitur **FOS** dan **GLCM** untuk memprediksi jenis penyakit paru-paru dari citra X-ray dada. Silakan unggah gambar X-ray untuk mendapatkan hasil prediksi.
""")

# Sidebar
st.sidebar.header("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar X-ray", type=["jpg", "jpeg", "png"])

# Fungsi Ekstraksi FOS + GLCM
@st.cache_data
def extract_fos_glcm(image):
    mask = np.ones_like(image)
    fos_feats, _ = pyfeats.fos(image, mask)
    glcm_feats = pyfeats.glcm_features(image)
    glcm_combined = np.concatenate([glcm_feats[0], glcm_feats[1]])
    drop = [12, 13, 14, 20, 26, 27, 28]
    glcm_filtered = np.delete(glcm_combined, drop)
    return fos_feats[:10], glcm_filtered

# Load model gabungan multiclass (4 kelas)
@st.cache_resource
def load_model_and_assets():
    model = joblib.load("model_svm.pkl")
    scaler = joblib.load("scaler.pkl")
    mask = joblib.load("best_mask_kombinasi.pkl")
    labels = ['Normal', 'Tuberculosis', 'Pneumonia', 'Covid19']
    return model, scaler, mask, labels

if uploaded_file is not None:
    # Preprocessing gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_clahe = clahe.apply(img_resized)

    # Tampilkan gambar
    st.subheader("📷 Gambar X-ray yang Diunggah")
    st.image(img_clahe, caption="Gambar X-ray setelah preprocessing (Grayscale + Resize + CLAHE)", use_column_width=True, channels="GRAY")

    # Ekstraksi fitur
    fos, glcm = extract_fos_glcm(img_clahe)
    combined = np.concatenate([fos, glcm])

    # Tampilkan fitur
    st.subheader("🧬 Ekstraksi Fitur FOS dan GLCM")
    df_feat = pd.DataFrame([np.concatenate([fos, glcm])])
    df_feat.columns = [
        'FOS_Mean', 'FOS_Std', 'FOS_Median', 'FOS_Mode', 'FOS_Skewness',
        'FOS_Kurtosis', 'FOS_Energy', 'FOS_Entropy', 'FOS_Min', 'FOS_Max'
    ] + [f"GLCM_{i}" for i in range(len(glcm))]
    st.dataframe(df_feat, use_container_width=True)

    # Prediksi
    model, scaler, mask, labels = load_model_and_assets()
    masked_feat = combined[mask]
    scaled_feat = scaler.transform([masked_feat])
    pred = model.predict(scaled_feat)[0]
    scores = model.decision_function(scaled_feat)

    st.subheader("📌 Hasil Prediksi")
    st.success(f"**Prediksi: {labels[pred]}**")

    # Visualisasi skor
    st.subheader("📈 Confidence per Kelas")
    scores_df = pd.DataFrame({
        'Kelas': labels,
        'Confidence': scores[0]
    })
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=scores_df, x='Kelas', y='Confidence', palette='coolwarm', ax=ax)
    ax.set_title("Confidence Score per Kelas")
    st.pyplot(fig)
