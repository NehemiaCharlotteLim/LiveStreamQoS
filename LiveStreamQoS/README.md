# LiveStreamQoS

Aplikasi penelitian dengan tiga menu: **Home**, **Mulai Monitoring**, dan **About**. Model XGBoost terbaik (`.joblib`) dari penelitian tetap digunakan tanpa pelatihan ulang. Kuesioner SUS di Google Form disiapkan secara terpisah oleh peneliti.

## Struktur

```
LiveStreamQoS/
  app.py
  wifi_agent.py
  start_wifi.bat
  component_wifi/index.html
  model_bundle_qos_4parameter_CV_TERBAIK.joblib
  requirements.txt
  README.md
```

## Deployment Streamlit Community Cloud

1. Ekstrak ZIP sekali. Di GitHub, tempatkan folder `LiveStreamQoS` **langsung di root repository**, tanpa membuat folder `LiveStreamQoS/LiveStreamQoS/` tambahan.
2. **Main file path:** `LiveStreamQoS/app.py`; branch `main`.
3. Disarankan Python 3.12. Dependensi tercantum dalam `requirements.txt`.
4. Buka alamat aplikasi online. Home dan About dapat digunakan tanpa penginstalan agen.

## Mengukur WiFi laptop (bukan WiFi server)

Karena Streamlit Cloud tidak dapat membaca adaptor WiFi pengunjung, jalankan **agen pengukur lokal** pada laptop yang ingin diukur.

**Windows:** pada halaman **Mulai Monitoring** klik tombol **Unduh Pengukur WiFi (Windows)**, ekstrak seluruh isi `LiveStreamQoS_WiFi.zip`, lalu klik dua kali `start_wifi.bat` (Python harus terpasang). Biarkan terminal agen tetap terbuka. Chrome mungkin meminta izin **akses jaringan lokal** ketika klik Mulai Monitoring pada website, pilih Izinkan.

Agen beroperasi melalui `http://127.0.0.1:8765` dan hanya melayani origin website yang diizinkan (`https://livestreamqos.streamlit.app` dan localhost). Jika aplikasi menggunakan subdomain Streamlit lain, jalankan `python wifi_agent.py --origin https://NAMA-APLIKASI.streamlit.app` untuk mengizinkannya secara eksplisit. Agen tidak terbuka ke internet dan tidak mengirim data otomatis ke server lain. Website membaca status melalui komponen browser dan meneruskannya ke model melalui sesi Streamlit.

**Jika browser memblokir akses agen**: ubah izin Local network / Loopback network pada Site settings untuk URL Streamlit, atau jalankan aplikasi secara lokal. Mekanisme akses browser-ke-localhost dapat berbeda menurut browser, versi dan kebijakan perangkat.

### Menjalankan aplikasi secara lokal

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Di jendela terminal lain, jalankan `python wifi_agent.py` (jika memakai Windows cukup `start_wifi.bat`).

### Batasan pengukuran

- **Latency, jitter, packet loss**: dari ICMP ping bila tersedia. Bila tidak tersedia, cadangan TCP menghasilkan waktu koneksi dan tingkat kegagalan TCP; tingkat kegagalan TCP **bukan** packet loss ICMP.
- **Throughput upload**: lalu lintas data terkirim adaptor per 5 detik, **bukan** kapasitas internet dari speed test.
- Hasil model adalah **indikasi QoS** Normal/Gangguan, bukan bukti langsung buffering video.
- Tidak ada simulasi yang disajikan sebagai hasil pengukuran WiFi nyata.
- Mengaktifkan agen dan memberikan izin jaringan lokal merupakan tindakan eksplisit dari pengguna.

## Pengujian sebelum SUS

Pastikan penguji dapat membuka Home, memulai dan menghentikan monitoring pada laptop pengujian, serta membuka About. Google Form SUS tetap terpisah; jangan gunakan hasil SUS sebagai ukuran akurasi model.
