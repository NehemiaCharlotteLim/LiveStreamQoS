LIVESTREAMQOS - UJI MODEL XGBOOST TERBAIK TANPA HTTP
====================================================

PERUBAHAN DARI VERSI LAMA
- Komponen monitoring browser, fetch HTTP, probe HTTP, timer, dan gagalnya
  12 sampel akibat "Failed to fetch" telah DIHAPUS dari versi ini.
- Jenis jaringan ditampilkan "WiFi" sebagai LABEL, bukan deteksi otomatis.
- Model XGBoost asli tetap berada dalam models/ dan dipakai melalui predict().
- Uji dua skenario dengan 1 klik: input QoS baik dan QoS menurun. Kedua
  skenario berisi DATA SINTETIS; hasil status ditentukan XGBoost asli.
- Input manual 12 sampel dan impor CSV berisi kolom:
  latency_ms,jitter_ms,packet_loss_percent,throughput_mbps
- Tanpa sumber data pengukuran, tidak ada monitoring kondisi WiFi nyata.

CARA DEPLOY KE STREAMLIT CLOUD
1. Ekstrak ZIP ini, lalu copy seluruh isinya ke ROOT repository GitHub.
2. Pastikan model joblib berukuran sekitar 1,77 MB berada di:
   LivestreamQoS_Monitoring(1)/models/model_bundle_qos_4parameter_CV_TERBAIK.joblib
3. Main file: LivestreamQoS_Monitoring(1)/app.py
4. Pilih Python 3.11 yang kompatibel dengan requirements.txt.
5. Deploy/reboot hingga aplikasi memuat. Bila menggunakan repo sebelumnya,
   file lama monitor_component/ dan static/ TIDAK diperlukan lagi.

CARA PENGUJIAN
- Tab "Uji Normal / Gangguan": klik "Jalankan Uji Normal dan Gangguan".
  Anda akan memperoleh dua hasil dari model asli, bukan aturan manual.
- Tab "Input Manual / CSV": ubah 12 sampel secara langsung atau unggah
  CSV dari sumber QoS eksternal, lalu klik "Prediksi Menggunakan XGBoost".
- Empat angka dalam setiap baris merupakan parameter QoS pada satu interval.
  Saat memakai contoh sintetis, 12 baris mewakili format 12 × 5 detik yang
  diperlukan model, BUKAN 60 detik pengukuran jaringan nyata.

METODE & BATASAN PENELITIAN
- File model asli tidak diubah atau dilatih ulang.
- Fitur: 24 ringkasan dari 12 × 4 data, menurut metadata model.
- Prediksi target 60 detik ke depan berasal dari data pelatihan model.
- Skenario sintetis hanya untuk demonstrasi fungsional model,
  BUKAN uji kinerja QoS koneksi, uji streaming aktual, atau validasi akurasi.
- Pengukuran nyata dapat diimpor melalui CSV dari alat eksternal yang sahih.
- Rumus std (ddof=1) dan delta (last-first) perlu diverifikasi dengan kode
  pelatihan asli. Keluaran XGBoost memang asli; validitas fitur masih perlu
  disandingkan dengan notebook training.
- Label data latih adalah proxy QoS empiris, bukan verifikasi kualitas
  playback video.
- File joblib adalah format pickle: muat hanya file tepercaya.

CARA MENJALANKAN LOKAL
Dari folder ZIP hasil ekstraksi:
python -m pip install -r requirements.txt
streamlit run LivestreamQoS_Monitoring(1)/app.py
