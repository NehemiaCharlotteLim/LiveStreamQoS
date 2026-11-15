# LiveStreamQoS: Pengujian XGBoost Tanpa HTTP

Aplikasi Streamlit untuk demonstrasi prediksi kelas **Normal / Gangguan (indikatif)** memakai model **XGBoost terbaik penelitian**. Pengujian ini **tidak** otomatis mengukur WiFi atau melakukan pengukuran HTTP.

## Struktur file dan deployment

Unggah **isi folder hasil ekstraksi** ke root repository GitHub `NehemiaCharlotteLim/LiveStreamQoS`, bukan ZIP-nya.

```text
LiveStreamQoS/                                [root repository]
├── .streamlit/config.toml
├── requirements.txt
├── README.md
└── LiveStreamQoS_Monitoring(1)/
    └── LiveStreamQoS_Cloud/
        ├── .streamlit/config.toml
        ├── app.py
        ├── inference.py
        ├── scenarios.py
        ├── requirements.txt
        └── models/
            └── model_bundle_qos_4parameter_CV_TERBAIK.joblib
```

Pengaturan Streamlit Community Cloud:

- Repository: `NehemiaCharlotteLim/LiveStreamQoS`
- Branch: `main`
- **Main file path: `LiveStreamQoS_Monitoring(1)/LiveStreamQoS_Cloud/app.py`**
- Jangan gunakan alamat web file GitHub dengan `/blob/main/` sebagai main file path; gunakan path relatif di atas atau Interactive Picker.
- Untuk aplikasi `livestreamqos.streamlit.app` yang sudah terdaftar, lakukan pembaruan/reboot aplikasi dari **My apps**, bukan membuat aplikasi baru dengan URL yang sama.

## Cara menggunakan

1. Buka aplikasi pada Streamlit.
2. Tab **Uji Normal / Gangguan**: klik **Jalankan Uji Normal dan Gangguan**. Dua skenario sintetis memberi input 12 sampel pada model asli; masing-masing kelas ditentukan oleh `model.predict()` dan dapat berubah bila model diganti.
3. Tab **Input Manual / CSV**: masukkan 12 baris × 4 parameter QoS (latency ms, jitter ms, packet loss %, throughput Mbps), kemudian klik prediksi. Untuk CSV lebih dari 12 baris, aplikasi mengambil 12 baris terakhir.
4. Label jaringan **WiFi** adalah teks tampilan sesuai permintaan; tidak ada deteksi adaptor WiFi otomatis.

## Informasi metodologi

- XGBoost dalam pipeline `SimpleImputer` + `XGBClassifier` dimuat dari file `.joblib` asli. Tidak ada training ulang dan tidak ada aturan ambang QoS manual untuk menampilkan kelas.
- Pipeline menerima 24 fitur statistik dari 12 interval riwayat (masing-masing dianggap 5 detik): `last`, `mean`, `min`, `max`, `std`, `delta` dari empat parameter QoS.
- Horizon model: 60 detik berikutnya, sesuai metadata bundle.
- Skenario bawaan adalah **data sintetis**, sehingga **bukan** pengukuran kondisi jaringan nyata atau bukti gangguan streaming.
- Untuk pengujian lapangan, isi CSV dari pengukuran jaringan yang benar dan samakan prosedur pengukuran dan rekayasa fitur dengan kode pelatihan penelitian.
- Asumsi perhitungan `std` adalah `ddof=1`; `delta` adalah sampel terakhir dikurangi pertama. Cocokkan kedua rumus tersebut dengan notebook pelatihan sebelum menarik kesimpulan akhir skripsi.
- `.joblib` adalah format pickle; muat hanya file yang Anda percayai.

## Jika deployment tidak berjalan

1. Pastikan `scenarios.py`, `inference.py`, dan model `.joblib` berada **sefolder relatif** seperti struktur di atas.
2. Pastikan main file path persis, termasuk kapitalisasi dan `(1)`.
3. Periksa **Manage app → Cloud logs** untuk melihat error Python/dependency.
4. Tombol **Deploy** yang nonaktif/abu-abu biasanya menandakan formulir belum valid atau alamat aplikasi sudah digunakan, bukan masalah kode Python yang sudah berjalan.
