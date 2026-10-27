import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="SPK Prediksi Diabetes KNN",
    page_icon="🩺",
    layout="wide"
)

# =========================
# DESAIN DAN WARNA
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #F8F5FF;
}

[data-testid="stSidebar"] {
    background-color: #E9DDFB;
}

h1, h2, h3 {
    color: #56358C;
}

.header-box {
    background: linear-gradient(120deg, #7952B3, #B38AE8);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

.header-box h1 {
    color: white;
    font-size: 32px;
}

.header-box p {
    color: #FFFFFF;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E4D7F7;
    padding: 18px;
    border-radius: 14px;
}

.stButton > button {
    background-color: #7952B3;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #5E3B91;
    color: white;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E4D7F7;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-box">
    <h1>🩺 SPK PREDIKSI DIABETES</h1>
    <p>Sistem Pendukung Keputusan Menggunakan
    Algoritma K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DATASET DAN MODEL
# =========================
@st.cache_data
def load_dataset():
    url = (
        "https://raw.githubusercontent.com/"
        "plotly/datasets/master/diabetes.csv"
    )
    df = pd.read_csv(url)

    # Mengganti nilai nol yang tidak masuk akal
    # pada beberapa variabel klinis dengan NaN.
    kolom_nol = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]

    for kolom in kolom_nol:
        df[kolom] = df[kolom].replace(0, np.nan)
        df[kolom] = df[kolom].fillna(df[kolom].median())

    return df


@st.cache_resource
def buat_model():
    df = load_dataset()

    fitur = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    X = df[fitur]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
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

    return model, akurasi, cm, df, fitur


try:
    model, akurasi, cm, df, fitur = buat_model()

    # =========================
    # MENU SIDEBAR
    # =========================
    st.sidebar.title("🪻 MENU APLIKASI")
    menu = st.sidebar.radio(
        "Pilih halaman:",
        [
            "Beranda",
            "Prediksi Diabetes",
            "Visualisasi Data",
            "Evaluasi Model",
            "Tentang Aplikasi"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("Metode: K-Nearest Neighbors")
    st.sidebar.caption("Tema: Lavender Purple")

    # =========================
    # BERANDA
    # =========================
    if menu == "Beranda":

        st.subheader("Dashboard Sistem")

        st.write(
            "Aplikasi ini digunakan untuk mempelajari "
            "klasifikasi diabetes berdasarkan data kesehatan "
            "dengan algoritma KNN."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Jumlah Dataset", f"{len(df)}")
        c2.metric("Jumlah Variabel", "8")
        c3.metric("Akurasi Pengujian", f"{akurasi * 100:.2f}%")

        st.divider()

        st.subheader("Informasi Dataset")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Pilih menu Prediksi Diabetes untuk memasukkan "
            "data dan melihat hasil klasifikasi model."
        )

    # =========================
    # PREDIKSI DIABETES
    # =========================
    elif menu == "Prediksi Diabetes":

        st.subheader("📝 Form Prediksi Diabetes")

        st.write(
            "Masukkan data sesuai variabel yang tersedia, "
            "kemudian tekan tombol prediksi."
        )

        with st.form("form_diabetes"):

            kiri, kanan = st.columns(2)

            with kiri:
                pregnancies = st.number_input(
                    "Jumlah Kehamilan",
                    min_value=0,
                    max_value=20,
                    value=1
                )

                glucose = st.number_input(
                    "Kadar Glukosa",
                    min_value=1,
                    max_value=300,
                    value=120
                )

                bp = st.number_input(
                    "Tekanan Darah Diastolik",
                    min_value=1,
                    max_value=200,
                    value=70
                )

                skin = st.number_input(
                    "Ketebalan Kulit",
                    min_value=0,
                    max_value=100,
                    value=20
                )

            with kanan:
                insulin = st.number_input(
                    "Kadar Insulin",
                    min_value=0,
                    max_value=900,
                    value=80
                )

                bmi = st.number_input(
                    "BMI",
                    min_value=1.0,
                    max_value=90.0,
                    value=25.0,
                    step=0.1
                )

                dpf = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5,
                    step=0.01
                )

                age = st.number_input(
                    "Umur",
                    min_value=1,
                    max_value=120,
                    value=25
                )

            proses = st.form_submit_button(
                "🔍 PROSES PREDIKSI"
            )

        if proses:

            data_input = pd.DataFrame(
                [[
                    pregnancies, glucose, bp, skin,
                    insulin, bmi, dpf, age
                ]],
                columns=fitur
            )

            hasil = int(model.predict(data_input)[0])
            proba = model.predict_proba(data_input)[0]

            st.divider()
            st.subheader("📊 Hasil Klasifikasi")

            kiri, kanan = st.columns(2)

            with kiri:
                if hasil == 1:
                    st.error(
                        "Hasil model: Kelas diabetes (1)"
                    )
                else:
                    st.success(
                        "Hasil model: Kelas non-diabetes (0)"
                    )

            with kanan:
                st.metric(
                    "Probabilitas kelas diabetes",
                    f"{proba[1] * 100:.2f}%"
                )

            st.write("*Rincian data yang diproses:*")
            st.dataframe(
                data_input,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "Hasil ini merupakan klasifikasi model "
                "untuk pembelajaran, bukan diagnosis medis. "
                "Silakan berkonsultasi dengan tenaga kesehatan "
                "untuk pemeriksaan diabetes."
            )

    # =========================
    # VISUALISASI DATA
    # =========================
    elif menu == "Visualisasi Data":

        st.subheader("📈 Visualisasi Dataset")

        kiri, kanan = st.columns(2)

        with kiri:
            st.write("Distribusi Kelas Diabetes")

            jumlah = df["Outcome"].value_counts().sort_index()

            fig, ax = plt.subplots()
            ax.bar(
                ["Non-diabetes", "Diabetes"],
                [
                    jumlah.get(0, 0),
                    jumlah.get(1, 0)
                ]
            )
            ax.set_ylabel("Jumlah data")
            ax.set_title("Distribusi Outcome")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        with kanan:
            st.write("Perbandingan Glukosa")

            fig, ax = plt.subplots()
            df.boxplot(
                column="Glucose",
                by="Outcome",
                ax=ax
            )
            ax.set_xlabel("Outcome (0 atau 1)")
            ax.set_ylabel("Glukosa")
            ax.set_title("Glukosa berdasarkan Outcome")
            fig.suptitle("")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Statistik Deskriptif")
        st.dataframe(
            df.describe().round(2),
            use_container_width=True
        )

    # =========================
    # EVALUASI MODEL
    # =========================
    elif menu == "Evaluasi Model":

        st.subheader("📋 Evaluasi Algoritma KNN")

        st.metric(
            "Akurasi Model",
            f"{akurasi * 100:.2f}%"
        )

        st.write("Confusion Matrix")

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
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("""
        Keterangan:

        - True Negative: data non-diabetes diprediksi
          non-diabetes.
        - False Positive: data non-diabetes diprediksi
          diabetes.
        - False Negative: data diabetes diprediksi
          non-diabetes.
        - True Positive: data diabetes diprediksi diabetes.
        """)

    # =========================
    # TENTANG APLIKASI
    # =========================
    elif menu == "Tentang Aplikasi":

        st.subheader("Tentang Sistem")

        st.write("""
        Sistem Pendukung Keputusan ini menerapkan metode
        K-Nearest Neighbors (KNN) untuk mengklasifikasikan
        data ke dalam dua kelas berdasarkan dataset diabetes.

        KNN bekerja dengan mencari data pelatihan yang paling
        dekat dengan data baru berdasarkan jarak antarvariabel.
        Model ini menggunakan lima tetangga terdekat (K = 5).

        Sebelum proses klasifikasi, data dinormalisasi
        menggunakan StandardScaler agar perbedaan skala
        variabel dapat diperhitungkan dengan lebih baik.
        """)

        st.latex(
            r"d(x,y)=\\sqrt{\\sum_{i=1}^{n}(x_i-y_i)^2}"
        )

        st.write(
            "*Sumber dataset:* Dataset diabetes publik "
            "untuk keperluan pembelajaran."
        )

except Exception as e:
    st.error(
        "Aplikasi gagal memuat data. "
        "Periksa koneksi internet dan konfigurasi dataset."
    )
    st.code(str(e))