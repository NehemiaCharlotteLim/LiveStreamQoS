"""LiveStreamQoS: monitoring browser responsive for laptops and phones.

All HTTP measurements run in the visitor's browser, not in Python on the server.
The classification is a transparent rule-based indicator, NOT the trained XGBoost model.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LiveStreamQoS", page_icon="📶", layout="centered")
HERE = Path(__file__).resolve().parent
st.title("LiveStreamQoS")
st.caption("Monitoring kondisi koneksi browser dan indikasi Normal/Gangguan")
home, pred, about = st.tabs(["Home", "Prediksi QoS", "About"])

with home:
    st.subheader("Selamat Datang")
    st.write("LiveStreamQoS merupakan aplikasi pemantauan kualitas koneksi dan indikasi gangguan jaringan. Versi online ini dapat dibuka melalui laptop maupun HP tanpa memasang sensor.")
    st.markdown("**Cara menggunakan:** buka tab **Prediksi QoS**, tekan **Mulai Monitoring**, dan amati nilai Latency, Jitter, Packet Loss, serta Throughput. Status **Normal/Gangguan** akan muncul setelah 12 pembacaan valid.")
    st.info("Pada website publik, pengukuran menggunakan permintaan HTTP dari browser ke server. Metode ini berbeda dari pengukuran ICMP serta throughput adaptor Wi-Fi yang digunakan untuk melatih model penelitian.")

with pred:
    st.subheader("Monitoring Wi-Fi dan Hasil Prediksi")
    st.caption("Klik Mulai Monitoring untuk mengukur koneksi perangkat yang sedang membuka website. Tidak memerlukan instalasi atau data contoh.")
    components.html((HERE/"browser_monitor.html").read_text(encoding="utf-8"),height=790,scrolling=False)

with about:
    st.subheader("Tentang LiveStreamQoS")
    st.write("Aplikasi ini memperkenalkan pemantauan kualitas koneksi melalui empat indikator: waktu respons, perubahan waktu respons, permintaan yang tidak berhasil, dan kecepatan transfer data.")
    st.markdown("**Kegunaan:** membantu pengguna memahami perubahan kondisi koneksi dan mencoba antarmuka monitoring melalui perangkat yang mereka miliki.")
    st.warning("Versi browser mengukur permintaan HTTP dan menyediakan indikator berbasis aturan. Hasilnya bukan klasifikasi model XGBoost penelitian, bukan pengukuran paket ICMP, dan bukan kepastian gangguan live streaming. Versi penelitian lokal tetap menggunakan model terlatih dan pengukuran adaptor Wi-Fi.")
