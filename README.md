# GlucoCheck — Streamlit KNN Diabetes Demo

Aplikasi dashboard Streamlit berbahasa Indonesia untuk demonstrasi klasifikasi menggunakan algoritma K-Nearest Neighbors (KNN). Desain dibuat dengan layout, warna, dan navigasi yang berbeda dari aplikasi referensi.

## Menjalankan di komputer

1. Ekstrak ZIP.
2. Buka folder `diabetes_knn_streamlit` di VS Code.
3. Pastikan Python 3.10 atau lebih baru terpasang.
4. Buka Terminal di folder proyek, lalu jalankan:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Browser akan membuka alamat lokal Streamlit (biasanya `http://localhost:8501`).

## Deploy ke Streamlit Community Cloud

1. Upload isi folder ini ke repository GitHub.
2. Buka Streamlit Community Cloud dan buat aplikasi baru.
3. Pilih repository, branch, dan file utama `app.py`.
4. Deploy.

## Isi folder

- `app.py` — aplikasi utama.
- `diabetes.csv` — dataset sintetis untuk demo.
- `requirements.txt` — daftar library.
- `README.md` — petunjuk menjalankan aplikasi.

## Penting

`diabetes.csv` adalah **data sintetis untuk demonstrasi pemrograman**, bukan data pasien nyata atau dataset klinis tervalidasi. Hasil prediksi tidak dapat digunakan untuk diagnosis atau keputusan pengobatan.
