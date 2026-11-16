# LiveStreamQoS

Aplikasi Streamlit online dengan 3 menu: **Home**, **Mulai Monitoring**, **About**. Menggunakan **model XGBoost terbaik asli** penelitian yang berada di `models/`.

## Struktur

```text
LiveStreamQoS/
├── app.py
├── inference.py
├── scenarios.py
├── wifi_agent.py
├── components/wifi_bridge/index.html
├── models/model_bundle_qos_4parameter_CV_TERBAIK.joblib
├── requirements.txt
└── README.md
```

## Deploy ke GitHub dan Streamlit Community Cloud

1. Ekstrak ZIP. Unggah **folder `LiveStreamQoS` dan semua isinya** langsung ke akar repository `NehemiaCharlotteLim/LiveStreamQoS`. Jangan membuat tambahan folder `LiveStreamQoS/LiveStreamQoS` secara tidak sengaja.
2. Di Streamlit Community Cloud, gunakan `NehemiaCharlotteLim/LiveStreamQoS`, branch `main`, **Main file path = `LiveStreamQoS/app.py`**.
3. Saat membuat deployment baru, pilih **Python 3.12** melalui *Advanced settings*. Perubahan versi Python pada aplikasi yang sudah deploy mengharuskan deploy ulang; menambahkan file `runtime.txt` saja tidak mengubah versi Python di Community Cloud.
4. Paket menggunakan **xgboost-cpu==3.1.3** untuk inferensi, bukan instalasi XGBoost dengan dependensi GPU NVIDIA. Semua dependensi aplikasi ada di folder yang sama dengan `app.py`.
5. Kode Home tidak memuat model hingga ada permintaan prediksi. Ini membantu startup awal.

## WiFi otomatis (laptop Windows)

Aplikasi Streamlit yang dihosting cloud **tidak bisa langsung membaca adaptor WiFi pengguna**. Untuk pengukuran WiFi otomatis dari laptop, jalankan agen lokal `wifi_agent.py` pada laptop **yang sama dengan browser** yang membuka aplikasi Streamlit online.

1. Pastikan Python sudah terpasang di laptop Windows.
2. Buka PowerShell pada direktori `LiveStreamQoS`, lalu jalankan:

   ```powershell
   py -m pip install psutil
   py wifi_agent.py --origin https://livestreamqos.streamlit.app
   ```

   Ganti `--origin` dengan URL aplikasi Streamlit Anda jika URL berbeda. Jika adaptor tidak terdeteksi, jalankan `py wifi_agent.py --origin https://livestreamqos.streamlit.app --interface "Wi-Fi"` dengan nama adaptor yang benar.

3. Biarkan terminal agen terbuka. Buka aplikasi online pada **Chrome di laptop yang sama**. Pilih **Mulai Monitoring → WiFi otomatis (laptop)**, lalu tekan **Mulai Monitoring**. Jika browser menanyakan izin akses jaringan lokal atau loopback, berikan izin untuk aplikasi Anda.
4. Agen mengumpulkan sampel valid tiap ±5 detik. Setelah 12 sampel valid tersedia, aplikasi meminta prediksi XGBoost dan memperbarui hasil saat sampel berubah.

**Batasan:** Jika browser memblokir koneksi loopback/iframe, monitoring otomatis mungkin tidak dapat berjalan meskipun agen aktif. Dalam hal ini, gunakan file `pengukuran_wifi.csv` yang disimpan otomatis oleh agen lalu unggah melalui pilihan **Unggah CSV**, atau lakukan uji lokal. HP tidak dapat mengakses agen localhost yang dijalankan pada laptop lain. Tidak ada janji bahwa seluruh browser dapat mengakses agen ini.

**Parameter agen:** latency dan packet loss didapat dari ping ICMP ke target (default `1.1.1.1`), jitter dihitung dari variasi ping berhasil, throughput dihitung dari selisih `bytes_recv` adaptor WiFi (laju transfer **teramati**, bukan kapasitas WiFi). Tidak ada unduh HTTP dari aplikasi cloud. Ketika pengukuran tidak lengkap, agen tidak mengarang nilai dan tidak memasukkan sampel invalid.

**Keamanan:** Agen hanya mendengarkan di `127.0.0.1` laptop pengguna, tidak di alamat publik, dan melayani statistik QoS tanpa SSID ataupun kredensial jaringan. Tidak perlu mengunggah password atau token. Akhiri agen dengan Ctrl+C.

## Pilihan lain tanpa agen

Menu **Mulai Monitoring** juga menyediakan: (a) uji dua skenario sintetis, (b) input manual 12 sampel, (c) unggah CSV yang berisi kolom `latency_ms,jitter_ms,packet_loss_percent,throughput_mbps` minimal 12 baris. Status Normal/Gangguan hanya dihasilkan `model.predict()` dari model asli. Contoh sintetis **bukan** hasil monitoring WiFi sebenarnya.

## Metodologi penelitian

Model bekerja pada 12 interval × 5 detik dan 24 fitur statistik (4 parameter × 6 statistik). Hasil adalah label **Normal QoS indikatif** atau **Penurunan QoS indikatif** selama horizon 60 detik ke depan, dan bukan bukti pasti gangguan video streaming. `std` memakai `ddof=1` dan `delta` adalah nilai akhir dikurangi nilai awal. Keduanya perlu dikonfirmasi terhadap notebook pelatihan agar rekayasa fitur identik.

File model `.joblib` adalah pickle; hanya gunakan bundle model tepercaya.
