import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="centered"
)

# ==========================
# TEMA BIRU UNGU
# ==========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eaf2ff, #eee7ff);
    color: #202d50;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #304ffe, #7950e8);
    padding: 32px;
    border-radius: 25px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px #5965ba35;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    font-size: 17px;
    color: #f0f1ff;
}

.panel {
    background: rgba(255,255,255,0.92);
    padding: 25px;
    border-radius: 22px;
    margin-bottom: 20px;
    box-shadow: 0 5px 20px #354b8b12;
}

h2, h3 {
    color: #3447a5;
}

.stButton > button {
    width: 100%;
    height: 55px;
    color: white;
    background: linear-gradient(90deg, #4058e8, #8555df);
    border: none;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #4058e8;
}

[data-testid="stMetric"] {
    background: #f1f3ff;
    padding: 16px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Analyzer</h1>
    <p>
        Sistem Pendukung Keputusan untuk memperkirakan
        risiko diabetes menggunakan algoritma
        K-Nearest Neighbors (KNN).
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================
# MEMBUAT MODEL KNN DEMO
# ==========================
@st.cache_resource
def train_model():
    rng = np.random.default_rng(21)
    n = 800

    age = rng.integers(20, 81, n)
    bmi = np.clip(rng.normal(27, 6, n), 15, 50)
    glucose = np.clip(rng.normal(115, 30, n), 60, 220)
    bp = np.clip(rng.normal(78, 12, n), 40, 130)
    family = rng.integers(0, 2, n)

    # Label buatan untuk demonstrasi, bukan data medis nyata.
    score = (
        (age >= 45) * 0.8
        + (bmi >= 30) * 0.8
        + (glucose >= 126) * 1.8
        + (bp >= 80) * 0.4
        + family * 0.8
        + rng.normal(0, 0.8, n)
    )

    y = (score >= 2.5).astype(int)

    X = pd.DataFrame({
        "Usia": age,
        "BMI": bmi,
        "Glukosa": glucose,
        "Tekanan Darah": bp,
        "Riwayat Keluarga": family
    })

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    model.fit(X, y)
    return model

model = train_model()

# ==========================
# INPUT DATA
# ==========================
st.markdown(
    '<div class="panel">',
    unsafe_allow_html=True
)

st.subheader("📋 Data Pasien")
st.write("Masukkan informasi berikut untuk menjalankan simulasi.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Usia (tahun)",
        min_value=1,
        max_value=120,
        value=35
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=24.5,
        step=0.1
    )

    glucose = st.number_input(
        "Glukosa",
        min_value=20,
        max_value=500,
        value=100
    )

with col2:
    bp = st.number_input(
        "Tekanan darah",
        min_value=20,
        max_value=250,
        value=75
    )

    family_text = st.selectbox(
        "Riwayat diabetes keluarga",
        ["Tidak ada", "Ada"]
    )

    family = 1 if family_text == "Ada" else 0

st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# PREDIKSI
# ==========================
if st.button("🔍 Analisis Risiko Diabetes"):

    patient = pd.DataFrame([{
        "Usia": age,
        "BMI": bmi,
        "Glukosa": glucose,
        "Tekanan Darah": bp,
        "Riwayat Keluarga": family
    }])

    prediction = int(model.predict(patient)[0])
    probability = model.predict_proba(patient)[0][1] * 100

    st.markdown("---")
    st.subheader("📊 Hasil Analisis")

    if prediction == 1:
        st.error("Hasil simulasi model: Risiko lebih tinggi")
    else:
        st.success("Hasil simulasi model: Risiko lebih rendah")

    st.metric(
        "Skor model untuk kelas risiko lebih tinggi",
        f"{probability:.1f}%"
    )

    st.progress(int(round(probability)))

    st.write("*Ringkasan data yang dimasukkan:*")
    st.dataframe(patient, use_container_width=True, hide_index=True)

    st.info(
        "Aplikasi ini menggunakan dataset sintetis untuk "
        "demonstrasi algoritma KNN. Hasil bukan diagnosis "
        "medis dan tidak menunjukkan probabilitas klinis "
        "yang tervalidasi. Untuk penilaian kesehatan, "
        "konsultasikan pemeriksaan dengan tenaga kesehatan."
    )

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption(
    "Diabetes Risk Analyzer | SPK dengan metode KNN")