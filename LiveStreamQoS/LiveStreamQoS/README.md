# LiveStreamQoS Monitoring

Aplikasi Streamlit dengan tiga menu: **Home**, **Mulai Monitoring**, dan **About**.

- **Home:** pengenalan aplikasi dan empat parameter QoS.
- **Mulai Monitoring:** uji Normal/Gangguan melalui skenario contoh, input manual 12 sampel, atau upload CSV.
- **About:** informasi program dan proses prediksi.

Model yang digunakan adalah **XGBoost terbaik asli penelitian**, tidak dilatih ulang. Keputusan Normal/Gangguan menggunakan `model.predict()` (pipeline). Tidak ada probe HTTP, dan aplikasi tidak mengukur WiFi secara otomatis. Data skenario merupakan contoh sintetis. Untuk pengujian nyata, masukkan hasil pengukuran melalui editor atau CSV.

## Struktur file

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

## Unggah ke GitHub

1. Ekstrak ZIP.
2. Unggah folder `LiveStreamQoS_Monitoring` lengkap ke root repository `NehemiaCharlotteLim/LiveStreamQoS`. Jangan unggah di dalam folder `LiveStreamQoS` lain.
3. Jika folder `LiveStreamQoS_Monitoring` sudah ada di GitHub, **ganti isi folder lama**, jangan membuat folder bersarang.
4. Commit ke branch `main`.

Pengaturan Streamlit Community Cloud:

- Repository: `NehemiaCharlotteLim/LiveStreamQoS`
- Branch: `main`
- Main file path: **`LiveStreamQoS_Monitoring/app.py`**
- Python: gunakan versi lingkungan yang kompatibel dengan dependensi.

Aplikasi yang sudah ter-deploy akan membaca perubahan di path-nya sendiri. Pastikan main file path deployment yang Anda gunakan sesuai lokasi kode baru.

## Format input CSV

Kolom wajib: `latency_ms`, `jitter_ms`, `packet_loss_percent`, dan `throughput_mbps`. Minimal 12 baris. Sistem menggunakan 12 baris terakhir. Nilai diurutkan dari pengukuran paling lama ke terbaru.

## Metode

Model mengolah empat parameter dalam 12 interval menjadi 24 fitur (`last`, `mean`, `min`, `max`, `std`, `delta`). Prediksi mengacu pada 60 detik ke depan menurut metadata bundle. Asumsi standar deviasi sampel `ddof=1` dan delta `nilai terakhir - nilai pertama` perlu dicocokkan dengan notebook pelatihan untuk evaluasi akademik akhir.

**Keamanan:** Muat hanya file `.joblib` Anda sendiri atau dari sumber tepercaya.
