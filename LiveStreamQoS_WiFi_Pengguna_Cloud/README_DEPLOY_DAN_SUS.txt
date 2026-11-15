LIVE STREAM QOS - WEBSITE CLOUD + SENSOR WIFI PENGGUNA (WINDOWS)

ISI PAKET
  app.py
  bridge.js
  model_bundle_qos_4parameter_CV_TERBAIK.joblib
  Sensor_WiFi_Windows.zip
  contoh_qos_1.csv / contoh_qos_2.csv
  requirements.txt
  wifi_sensor_local.py dan JALANKAN_SENSOR_WINDOWS.bat (sumber sensor)

DEPLOY KE STREAMLIT COMMUNITY CLOUD
1. Buat repository GitHub baru dan unggah semua file dalam folder ini (bukan folder pembungkus).
2. Masuk https://share.streamlit.io/ dan buat aplikasi dari GitHub.
3. Pilih branch main, app.py, Python 3.11 atau yang kompatibel dengan requirements.
4. Tunggu instalasi dependencies: Streamlit versi baru untuk komponen v2, scikit-learn 1.8.0, xgboost 3.1.3.
5. Bagikan URL https://NAMA.streamlit.app ke responden.

INSTRUKSI RESPONDEN (LAPTOP WINDOWS + CHROME/EDGE)
1. Buka link aplikasi, tab Prediksi QoS, opsi Wi-Fi laptop saya.
2. Unduh ZIP sensor yang disediakan di website.
3. Ekstrak, klik dua kali JALANKAN_SENSOR_WINDOWS.bat (membutuhkan Python yang terpasang).
4. Jendela CMD menampilkan kode penghubung sekali pakai. Salin ke website.
5. Klik Hubungkan Sensor Wi-Fi, lalu izinkan browser mengakses localhost / jaringan lokal bila diminta.
6. Tunggu sekitar 1 menit untuk 12 interval valid; nilai dan prediksi akan muncul.
7. Lanjutkan pengisian Google Forms SUS yang dibagikan peneliti secara terpisah.

TROUBLESHOOTING
- Sensor tidak terbuka: instal Python Windows dan pastikan pip/psutil tersedia.
- Tidak tersambung: gunakan Chrome/Edge terbaru pada laptop yang SAMA. Beri izin Site Settings > Local network access/Loopback network.
- Gagal karena kode: salin kode yang ditampilkan di jendela CMD saat ini.
- Ping tidak berhasil: pastikan ICMP ke 1.1.1.1 tidak diblokir. Sensor tidak mengganti RTT/Jitter dengan angka palsu.
- Website tidak terhubung karena kebijakan browser: pilih opsi Unggah CSV, pilih `riwayat_qos_sensor.csv` dari folder sensor.
- Jika memakai HP saja: tidak dapat menjalankan sensor Windows; bisa menggunakan contoh data, tetapi itu bukan monitoring Wi-Fi HP.
- Jika dibuka di custom domain (bukan *.streamlit.app): whitelist origin di fungsi authorized_origin sensor harus diperbarui secara aman.

KEAMANAN/PRIVASI
- Sensor hanya terikat ke 127.0.0.1, port 8765. Tidak dapat diakses dari jaringan rumah atau internet.
- Data sensor memerlukan token sementara dan izin origin browser.
- Cloud hanya menerima data QoS dari browser setelah tombol Hubungkan diklik, untuk melakukan inferensi, tanpa database terpusat.
- File CSV mentah disimpan di laptop responden.
- Tidak ada pengumpulan kuesioner SUS dalam website. Kuesioner dilakukan secara terpisah.

METODOLOGI
Empat parameter diambil dengan cara sama seperti pengumpulan dataset (ICMP dan counter bytes_sent adaptor).
Model memprediksi label indikatif QoS, bukan label ground truth gangguan playback.
Keberhasilan deployment web tidak menjamin sensor berfungsi pada setiap browser tanpa izin jaringan lokal.
