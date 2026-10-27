import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="GlucoCheck | Prediksi Diabetes", page_icon="🩺", layout="wide")

# Custom styling: clean teal/navy dashboard, intentionally different from a default Streamlit layout.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f3f8fb 0%, #f7fbf9 60%, #eef7f5 100%); color: #183447; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px; }
.hero { background: linear-gradient(115deg,#103b50,#176b70); padding: 30px 34px; border-radius: 24px; color: white; box-shadow: 0 14px 35px rgba(16,59,80,.15); }
.hero h1 { font-family: 'Manrope',sans-serif; color:white; font-size: 2.25rem; margin: 0 0 8px 0; }
.hero p { color:#d8eeee; margin:0; font-size:1.02rem; }
.pill { display:inline-block; background:#d9f5ed; color:#14665f; padding:6px 12px; border-radius:999px; font-weight:700; font-size:.78rem; margin-bottom:12px; }
.metric-card { background:white; border:1px solid #e2ecef; border-radius:18px; padding:18px 20px; box-shadow:0 5px 18px rgba(28,65,78,.04); min-height:112px; }
.metric-label { color:#6b8290; font-size:.85rem; font-weight:600; }
.metric-value { color:#153d50; font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; margin-top:4px; }
.section-title { font-family:'Manrope',sans-serif; color:#173f52; font-size:1.25rem; font-weight:800; margin: 4px 0 12px; }
.info-box { background:#e9f6f3; border-left:4px solid #1b8a80; border-radius:12px; padding:14px 16px; color:#244e55; }
.result-good { background:#e5f7ef; border:1px solid #a9e4cb; border-radius:18px; padding:22px; color:#155c43; }
.result-warn { background:#fff3e7; border:1px solid #ffd3a6; border-radius:18px; padding:22px; color:#854a12; }
footer {visibility:hidden;}
[data-testid="stSidebar"] { background:#123e50; }
[data-testid="stSidebar"] * { color:#f1fbfc !important; }
.stButton>button { background:#167c78; color:white; border:0; border-radius:12px; padding:.65rem 1rem; font-weight:700; }
.stButton>button:hover { background:#105e61; color:white; border:0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = Path(__file__).parent / "diabetes.csv"
    data = pd.read_csv(path)
    return data

@st.cache_resource
def train_model(k):
    data = load_data()
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, accuracy_score(y_test, pred), confusion_matrix(y_test, pred), data

data = load_data()
st.markdown("""
<div class="hero">
  <div class="pill">MACHINE LEARNING • K-NEAREST NEIGHBORS</div>
  <h1>GlucoCheck</h1>
  <p>Dashboard edukasi untuk memahami prediksi risiko diabetes menggunakan algoritma KNN.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("## 🩺 GlucoCheck")
    st.caption("Diabetes Prediction Studio")
    st.markdown("---")
    page = st.radio("NAVIGASI", ["Prediksi", "Eksplorasi Data", "Tentang Model"])
    st.markdown("---")
    k_value = st.slider("Jumlah tetangga (K)", min_value=3, max_value=15, value=5, step=2)
    st.caption("Nilai K ganjil membantu mengurangi kemungkinan hasil seri.")

model, accuracy, cm, data = train_model(k_value)
positive_rate = data["Outcome"].mean() * 100
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Jumlah data demo", f"{len(data):,}"),
    (c2, "Fitur input", "8"),
    (c3, "Akurasi uji model", f"{accuracy*100:.1f}%"),
    (c4, "Label positif pada data", f"{positive_rate:.1f}%"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
st.write("")

if page == "Prediksi":
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.markdown('<div class="section-title">Masukkan data pemeriksaan</div>', unsafe_allow_html=True)
        st.markdown("Isi nilai sesuai data yang tersedia. Angka awal hanya contoh dan dapat diubah.")
        with st.form("prediction_form"):
            a, b = st.columns(2)
            with a:
                pregnancies = st.number_input("Jumlah kehamilan", 0, 20, 2)
                glucose = st.number_input("Glukosa (mg/dL)", 0.0, 250.0, 120.0, step=1.0)
                bp = st.number_input("Tekanan darah diastolik (mmHg)", 0.0, 150.0, 72.0, step=1.0)
                skin = st.number_input("Ketebalan lipatan kulit (mm)", 0.0, 100.0, 25.0, step=1.0)
            with b:
                insulin = st.number_input("Insulin (µU/mL)", 0.0, 900.0, 85.0, step=1.0)
                bmi = st.number_input("BMI", 0.0, 80.0, 28.5, step=0.1)
                dpf = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.45, step=0.01)
                age = st.number_input("Usia (tahun)", 1, 120, 35)
            submitted = st.form_submit_button("🔎 Analisis data", use_container_width=True)
        if submitted:
            row = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
                "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }])
            prediction = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1]) * 100
            st.session_state["prediction"] = (prediction, probability, row)
    with right:
        st.markdown('<div class="section-title">Hasil analisis</div>', unsafe_allow_html=True)
        if "prediction" not in st.session_state:
            st.markdown('<div class="info-box"><b>Siap untuk dianalisis?</b><br>Masukkan data di sebelah kiri, lalu tekan tombol Analisis data untuk melihat hasil model KNN.</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("#### Cara membaca hasil")
            st.write("• Model membandingkan input dengan data yang dipakai saat pelatihan.")
            st.write("• Persentase merupakan keluaran model, bukan kepastian medis.")
            st.write("• Hasil dipengaruhi oleh kualitas dan keterbatasan data demo.")
        else:
            prediction, probability, row = st.session_state["prediction"]
            if prediction == 1:
                st.markdown(f'<div class="result-warn"><h3>Perlu perhatian lebih</h3><p>Model mengelompokkan input ke kelas positif pada data demo.</p><h2>{probability:.1f}%</h2><p>Skor probabilitas model</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-good"><h3>Kelas negatif menurut model</h3><p>Model mengelompokkan input ke kelas negatif pada data demo.</p><h2>{100-probability:.1f}%</h2><p>Skor kelas negatif dari model</p></div>', unsafe_allow_html=True)
            st.write("")
            st.caption("Ini bukan diagnosis. Konsultasikan hasil pemeriksaan dengan tenaga kesehatan.")
            with st.expander("Lihat ringkasan input"):
                st.dataframe(row, use_container_width=True, hide_index=True)
    st.write("")
    st.markdown('<div class="info-box"><b>Catatan penting:</b> dataset yang disertakan adalah data sintetis untuk demonstrasi pemrograman, bukan rekam medis nyata. Aplikasi ini tidak boleh digunakan untuk mendiagnosis atau menentukan pengobatan.</div>', unsafe_allow_html=True)

elif page == "Eksplorasi Data":
    st.markdown('<div class="section-title">Eksplorasi dataset</div>', unsafe_allow_html=True)
    l, r = st.columns([1.1, .9])
    with l:
        st.markdown("#### Cuplikan data")
        st.dataframe(data.head(15), use_container_width=True, hide_index=True)
    with r:
        st.markdown("#### Distribusi label")
        counts = data["Outcome"].value_counts().rename(index={0:"Kelas 0",1:"Kelas 1"}).sort_index()
        st.bar_chart(counts)
        st.markdown("#### Statistik ringkas")
        st.dataframe(data.describe().T.round(2), use_container_width=True)
    st.caption("Dataset pada ZIP ini dibuat secara sintetis untuk demo. Jangan ditafsirkan sebagai statistik populasi.")

else:
    st.markdown('<div class="section-title">Tentang aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Algoritma K-Nearest Neighbors (KNN)
    KNN mengelompokkan sebuah data berdasarkan kelas dari sejumlah tetangga terdekatnya.

    **Alur aplikasi**
    1. Membaca dataset demo dari `diabetes.csv`.
    2. Membagi data menjadi data latih dan data uji.
    3. Mengisi nilai kosong jika ada, lalu menstandarkan skala fitur.
    4. Melatih model KNN dan mengevaluasi akurasi pada data uji.
    5. Menggunakan model untuk memprediksi input dari formulir.

    **Fitur yang digunakan**
    - Jumlah kehamilan
    - Glukosa
    - Tekanan darah
    - Ketebalan lipatan kulit
    - Insulin
    - BMI
    - Diabetes pedigree function
    - Usia

    **Batasan:** dataset sintetis dibuat untuk demonstrasi alur machine learning. Akurasi pada data ini tidak menggambarkan kinerja klinis di dunia nyata.
    """)
    st.markdown("#### Confusion matrix pada data uji")
    st.dataframe(pd.DataFrame(cm, index=["Aktual 0", "Aktual 1"], columns=["Prediksi 0", "Prediksi 1"]), use_container_width=True)

st.markdown("---")
st.caption("GlucoCheck • Proyek pembelajaran Data Mining • KNN • Data sintetis untuk demonstrasi")
