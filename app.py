import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Prediksi Diabetes KNN",
    page_icon="🩺",
    layout="wide"
)

# Tampilan aplikasi
st.markdown("""
<style>
.stApp {
    background-color: #f3f7fb;
}
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #123b63;
}
.subtitle {
    color: #60758a;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce6ef;
}
.stButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Sistem Prediksi Risiko Diabetes '
    'Menggunakan Algoritma K-Nearest Neighbors (KNN)</div>',
    unsafe_allow_html=True
)

st.divider()

# Memuat dataset
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/"
        "plotly/datasets/master/diabetes.csv"
    )
    return pd.read_csv(url)

@st.cache_resource
def train_model():
    data = load_data()

    fitur = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ]

    X = data[fitur]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)
    akurasi = accuracy_score(y_test, prediksi)

    cm = confusion_matrix(y_test, prediksi)

    return model, akurasi, cm, data, fitur

try:
    model, akurasi, cm, data, fitur = train_model()

    # Ringkasan dashboard
    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", f"{len(data)}")
    col2.metric("Jumlah Fitur", "8")
    col3.metric("Akurasi Pengujian", f"{akurasi * 100:.2f}%")

    st.divider()

    # Menu
    menu = st.sidebar.radio(
        "MENU UTAMA",
        ["Prediksi Diabetes", "Visualisasi Data", "Tentang KNN"]
    )

    if menu == "Prediksi Diabetes":

        st.subheader("Masukkan Data Kesehatan")
        st.write(
            "Isi seluruh data berikut untuk menjalankan prediksi."
        )

        with st.form("form_prediksi"):

            c1, c2 = st.columns(2)

            with c1:
                pregnancies = st.number_input(
                    "Jumlah Kehamilan",
                    min_value=0, max_value=20, value=1
                )

                glucose = st.number_input(
                    "Kadar Glukosa",
                    min_value=0, max_value=300, value=120
                )

                blood_pressure = st.number_input(
                    "Tekanan Darah Diastolik",
                    min_value=0, max_value=200, value=70
                )

                skin = st.number_input(
                    "Ketebalan Kulit",
                    min_value=0, max_value=100, value=20
                )

            with c2:
                insulin = st.number_input(
                    "Insulin",
                    min_value=0, max_value=900, value=80
                )

                bmi = st.number_input(
                    "BMI",
                    min_value=0.0, max_value=90.0,
                    value=25.0, step=0.1
                )

                dpf = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0, max_value=3.0,
                    value=0.5, step=0.01
                )

                age = st.number_input(
                    "Umur",
                    min_value=1, max_value=120, value=25
                )

            submit = st.form_submit_button("🔍 Prediksi Sekarang")

        if submit:

            nilai = np.array([[
                pregnancies, glucose, blood_pressure,
                skin, insulin, bmi, dpf, age
            ]])

            hasil = model.predict(nilai)[0]
            probabilitas = model.predict_proba(nilai)[0]

            st.divider()
            st.subheader("Hasil Prediksi")

            a, b = st.columns(2)

            with a:
                if hasil == 1:
                    st.error(
                        "Hasil model: Terklasifikasi pada kelompok "
                        "diabetes."
                    )
                else:
                    st.success(
                        "Hasil model: Terklasifikasi pada kelompok "
                        "non-diabetes."
                    )

            with b:
                st.metric(
                    "Probabilitas Kelas Diabetes dari Model",
                    f"{probabilitas[1] * 100:.2f}%"
                )

            st.caption(
                "Probabilitas ini merupakan keluaran model, "
                "bukan kepastian seseorang menderita diabetes."
            )

            st.subheader("Data yang Dimasukkan")

            hasil_data = pd.DataFrame({
                "Variabel": fitur,
                "Nilai": nilai[0]
            })

            st.dataframe(
                hasil_data,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "Aplikasi ini hanya untuk pembelajaran dan "
                "skrining awal. Hasilnya tidak menggantikan "
                "pemeriksaan laboratorium atau diagnosis dokter."
            )

    elif menu == "Visualisasi Data":

        st.subheader("Visualisasi Dataset Diabetes")

        kiri, kanan = st.columns(2)

        with kiri:
            st.write("*Distribusi Status Outcome*")

            jumlah = data["Outcome"].value_counts().sort_index()

            fig, ax = plt.subplots()
            ax.bar(
                ["Non-diabetes (0)", "Diabetes (1)"],
                [jumlah.get(0, 0), jumlah.get(1, 0)]
            )
            ax.set_ylabel("Jumlah Data")
            ax.set_title("Distribusi Kelas")
            st.pyplot(fig)
            plt.close(fig)

        with kanan:
            st.write("*Perbandingan Glukosa*")

            fig, ax = plt.subplots()
            data.boxplot(
                column="Glucose",
                by="Outcome",
                ax=ax
            )
            ax.set_title("Glukosa berdasarkan Outcome")
            ax.set_xlabel("Outcome (0 = Non-diabetes, 1 = Diabetes)")
            ax.set_ylabel("Glukosa")
            fig.suptitle("")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Data Dataset")
        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prediksi 0", "Prediksi 1"])
        ax.set_yticklabels(["Aktual 0", "Aktual 1"])
        ax.set_xlabel("Hasil Prediksi")
        ax.set_ylabel("Data Aktual")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center"
                )

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.subheader("Tentang Algoritma KNN")

        st.write("""
        K-Nearest Neighbors (KNN) adalah algoritma machine
        learning yang menentukan kelas suatu data berdasarkan
        sejumlah tetangga terdekatnya.

        Pada aplikasi ini, model menggunakan 8 variabel:
        jumlah kehamilan, glukosa, tekanan darah, ketebalan kulit,
        insulin, BMI, Diabetes Pedigree Function, dan umur.

        Data dinormalisasi menggunakan StandardScaler agar
        perbedaan skala antarvariabel tidak mendominasi
        perhitungan jarak. Model menggunakan K = 5 tetangga
        terdekat dan dievaluasi menggunakan data pengujian.
        """)

        st.write("*Rumus jarak Euclidean:*")

        st.latex(
            r"d(x,y)=\\sqrt{\\sum_{i=1}^{n}(x_i-y_i)^2}"
        )

        st.write(
            f"Akurasi pada data pengujian: {akurasi * 100:.2f}%."
        )

except Exception as e:
    st.error("Aplikasi gagal memuat dataset atau model.")
    st.write("Periksa koneksi internet dan dataset.")
    st.code(str(e))