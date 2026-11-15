LIVESTREAMQOS - MONITORING DENGAN MODEL XGBOOST TERBAIK PENELITIAN
====================================================================

ISI
- LiveStreamQoS_Cloud/app.py: website Streamlit untuk HP dan laptop.
- LiveStreamQoS_Cloud/inference.py: konversi 12 × 4 data menjadi 24 fitur,
  lalu inferensi model asli; bukan klasifikasi ambang manual.
- LiveStreamQoS_Cloud/models/model_bundle_qos_4parameter_CV_TERBAIK.joblib:
  salinan utuh file model yang Anda unggah, tidak dilatih ulang.
- requirements.txt: versi paket yang cocok dengan metadata model.

CARA MENJALANKAN LOKAL
1. Gunakan Python 3.11 (atau versi kompatibel dengan dependensi).
2. Dari folder hasil ekstraksi:
      python -m pip install -r requirements.txt
      streamlit run LiveStreamQoS_Cloud/app.py
3. Buka tab Prediksi QoS, klik Mulai Monitoring.
4. Setelah 12 pengukuran HTTP valid (sekitar 60 detik), akan muncul prediksi
   NORMAL atau GANGGUAN (indikatif) yang dihitung model XGBoost.

CARA DEPLOY STREAMLIT COMMUNITY CLOUD
1. Ekstrak ZIP lalu unggah seluruh isi folder hasil ekstraksi ke repository
   GitHub, termasuk file .joblib di subfolder models dan folder .streamlit
   pada ROOT repository (aktifkan staticServing).
2. Pilih main file: LiveStreamQoS_Cloud/app.py
3. Pilih versi Python yang kompatibel dengan paket (disarankan Python 3.11).
4. Jalankan app. Jangan hanya mengunggah ZIP sebagai satu file.

FAKTA MODEL DIAMBIL DARI FILE JOBLIB ANDA
- best_model_name: XGBoost
- Pipeline: SimpleImputer + XGBClassifier
- 24 fitur = last, mean, min, max, std, delta x 4 parameter
- history_intervals: 12; interval_seconds: 5; horizon_intervals: 12
- kelas 0 = Normal QoS indikatif; 1 = Penurunan QoS indikatif
- label_mode: qos_proxy; label berasal dari label_biner_qos_indikatif
- model dilatih pada Dataset_QoS_Gabungan_1920_Baris.csv
- Seleksi: 4-fold session-wise purged forward-chaining cross-validation

BATASAN PENTING UNTUK PENELITIAN
- Status pada aplikasi adalah hasil model XGBoost asli, bukan hasil aturan
  threshold manual pada browser. Namun label training model merupakan
  QoS indikatif berdasarkan proxy empiris, BUKAN bukti gangguan streaming.
- Browser mengukur waktu respons HTTP; packet loss adalah persentase request
  HTTP gagal, bukan packet loss ICMP/IP; throughput adalah estimasi unduh HTTP.
  Kesesuaian distribusi data HTTP dengan data latih harus dievaluasi.
- Implementasi 24 fitur di aplikasi mengasumsikan std sampel (ddof=1) dan
  delta=nilai terakhir - nilai pertama. Bundle tidak menyimpan rumus detail;
  KONFIRMASIKAN dengan notebook pembentukan fitur penelitian. Sampai
  konfirmasi, konsistensi feature engineering BELUM terverifikasi penuh.
- Interval browser ±5 detik, tetapi dapat lebih panjang pada koneksi lambat.
- Aplikasi tidak merekam atau memverifikasi kondisi pemutaran video di HP.
- Hanya muat model joblib/pickle yang Anda percayai.

CATATAN
Jika fungsi pembuatan fitur penelitian memakai ddof=0 atau rumus delta yang
berbeda, ubah bagian `make_feature_frame` di inference.py agar persis sama
sebelum memakai hasil sebagai bukti penelitian.
