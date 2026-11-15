# LiveStreamQoS_Monitoring

Aplikasi Streamlit untuk **uji klasifikasi QoS Normal / Gangguan dengan model XGBoost terbaik asli**.

## Struktur ZIP

```text
LiveStreamQoS_Monitoring/
├── app.py
├── inference.py
├── scenarios.py
├── requirements.txt
├── README.md
└── models/
    └── model_bundle_qos_4parameter_CV_TERBAIK.joblib
```

Folder ini merupakan **satu-satunya folder utama** dalam ZIP. Tidak ada folder aplikasi bertingkat seperti pada paket sebelumnya.

## Langkah unggah ke GitHub

1. Ekstrak ZIP di komputer.
2. Buka root repository `NehemiaCharlotteLim/LiveStreamQoS` (jangan masuk ke folder `LiveStreamQoS` lama).
3. Unggah folder `LiveStreamQoS_Monitoring` **beserta seluruh isinya** ke root repository, lalu commit ke branch `main`.
4. Periksa GitHub hingga terlihat file berikut: `LiveStreamQoS_Monitoring/app.py`.
5. Saat deployment Streamlit, gunakan:
   - Repository: `NehemiaCharlotteLim/LiveStreamQoS`
   - Branch: `main`
   - **Main file path: `LiveStreamQoS_Monitoring/app.py`**
   - **Python version: 3.12** (pilih dari Advanced settings saat membuat deployment baru).
6. Jika aplikasi lama masih menunjuk ke `LiveStreamQoS/LiveStreamQoS/LiveStreamQoS_Cloud/app.py`, aplikasi lama **tidak otomatis berpindah** ke file baru. Ubah entrypoint lewat pengaturan jika tersedia, atau buat deployment baru dengan path di atas dan URL berbeda (mis. `livestreamqos-monitoring-uji`).

**Penting:** Anda tidak perlu membuat folder `LiveStreamQoS` tambahan di dalam ZIP ini. Folder repository GitHub sudah bernama `LiveStreamQoS`.

## Penggunaan

- **Uji Normal dan Gangguan:** menjalankan dua skenario sintetis, masing-masing berisi 12 sampel; hasil klasifikasi diputuskan oleh `.predict()` dari model XGBoost asli.
- **Input Manual / CSV:** masukkan 12 sampel empat parameter QoS (latency, jitter, packet loss, throughput).
- **Info dan Diagnostik:** melihat definisi label dan versi dependensi.

Tidak ada pengukuran HTTP. Label WiFi hanya label antarmuka, bukan identifikasi adaptor pengguna. Model menggunakan indikasi QoS, tidak memverifikasi buffering atau pemutaran streaming.

## Dependensi dan kestabilan

File `requirements.txt` ada **sefolder dengan `app.py`**, sesuai aturan Streamlit Community Cloud sehingga diutamakan dibanding `requirements.txt` lama pada root repository. Streamlit versi baru dan PyArrow 24 dipakai untuk mengurangi risiko crash startup PyArrow 25 yang dilaporkan pada Streamlit lama. Versi XGBoost 3.1.3 serta scikit-learn 1.8.0 sesuai metadata model.

**Pengujian yang dilakukan:** validasi sintaks Python, integritas ZIP dan file model, kesesuaian 24 fitur, serta pengujian `.predict()`/`.predict_proba()` untuk dua skenario. Deployment Streamlit Cloud sesungguhnya belum dapat diuji dari lingkungan pembuat file ini. Jika halaman tetap blank dengan Cloud logs kosong, periksa status layanan Streamlit dan Python version; masalah tersebut belum tentu berasal dari kode aplikasi.

## Catatan penelitian

Formula `std` memakai `ddof=1` dan `delta` = sampel terakhir dikurangi pertama. Pastikan kedua definisi tersebut sama dengan notebook pelatihan sebelum mengklaim keakuratan integrasi penelitian.

File `.joblib` memuat objek Python melalui pickle. Jangan mengganti file dengan model dari pihak yang tidak dipercaya.
