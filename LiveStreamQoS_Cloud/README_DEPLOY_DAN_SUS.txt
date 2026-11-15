LiveStreamQoS - VERSI HP TANPA INSTALASI + WINDOWS DENGAN SENSOR

AKSES WEBSITE
https://livestreamqos.streamlit.app

UNTUK PENGGUNA HP (ANDROID / IPHONE)
1. Buka alamat website dari browser HP.
2. Pilih Prediksi QoS > HP / Browser (langsung, tanpa instalasi).
3. Tekan Mulai Monitoring HP.
4. Nilai HTTP dan status indikatif Normal/Gangguan ditampilkan.
5. Pembacaan 12 kali kira-kira membutuhkan 1 menit. Bisa disimpan ke CSV.
6. Isi kuesioner SUS melalui link terpisah dari peneliti.

PENTING: HP tidak menghasilkan empat parameter model XGBoost.
Status HP memakai aturan eksploratif pada metrik HTTP browser, BUKAN
prediksi model XGBoost terlatih dan BUKAN bukti gangguan streaming.
Kualitas server hosting atau operator juga memengaruhi angka.
Data internet yang digunakan kira-kira 3MB per menit monitoring.

UNTUK WINDOWS
Pilih Laptop Windows (sensor lokal), unduh sensor, ekstrak dan jalankan
MULAI_MONITORING.bat. Website akan dibuka otomatis. Model XGBoost yang
sama digunakan untuk 24 fitur dari 4 parameter QoS.

UNTUK UPDATE GITHUB DAN STREAMLIT
1. Pastikan repository GitHub LiveStreamQoS berisi file app.py, browser_probe.js,
   bridge.js, model_bundle_qos_4parameter_CV_TERBAIK.joblib,
   Sensor_WiFi_Windows.zip, requirements.txt dan folder berikut:
   .streamlit/config.toml
   static/probe.bin
   static/probe.json
2. Jangan mengunggah ZIP utama sebagai satu-satunya file; ekstrak isi ZIP.
3. Upload/commit ke branch main. Pastikan folder .streamlit dan static ikut terunggah.
4. Streamlit Community Cloud deploy ulang otomatis.
5. Tes dari HP: menu HP/Browser > Mulai Monitoring HP. Bila tampil pesan
   berkas uji tidak dapat diakses, cek bahwa .streamlit/config.toml telah
   diunggah dan berisi [server] enableStaticServing = true.
6. Tidak perlu membuat Google Forms dalam website. SUS terpisah.

PRIVASI
Browser mengirim permintaan HTTP untuk mengukur koneksi dari perangkat
pengguna ke server Streamlit. Nilai metrik diproses di browser, tidak
dikirim ke backend model. File CSV hanya diunduh bila pengguna memilih.

PENGUJIAN
Syntax Python dan JavaScript diperiksa pada lingkungan pembuatan.
Pengujian langsung Safari/Chrome HP dan Streamlit Community Cloud perlu
Anda lakukan setelah deployment sebelum menyebarkan link untuk SUS.
