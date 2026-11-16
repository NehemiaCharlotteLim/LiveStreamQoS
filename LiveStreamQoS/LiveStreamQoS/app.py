"""LiveStreamQoS - tiga menu: Home, Mulai Monitoring, dan About.
Model XGBoost asli melakukan seluruh klasifikasi Normal/Gangguan.
Data disediakan melalui simulasi, editor manual, atau unggahan CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from inference import SOURCE_COLUMNS, WINDOW_SIZE, load_model_bundle, predict_window
from scenarios import create_scenario, to_dataframe

st.set_page_config(
    page_title="LiveStreamQoS | Monitoring QoS WiFi",
    page_icon="📶",
    layout="centered",
)

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "models" / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"


@st.cache_resource(show_spinner=False)
def get_model(path: str, mtime_ns: int):
    return load_model_bundle(Path(path))


try:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("File model XGBoost tidak ditemukan di folder models/.")
    model, bundle = get_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
except Exception as exc:
    st.error("Model XGBoost belum dapat dimuat. Periksa file model dan requirements.txt.")
    st.exception(exc)
    st.stop()


def infer(rows):
    return predict_window(model, bundle, rows)


def render_result(title: str, result: dict, samples=None):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        status = "NORMAL" if result["raw_class"] == 0 else "GANGGUAN"
        if result["raw_class"] == 0:
            st.success(f"Hasil prediksi XGBoost: **{status}**")
        else:
            st.error(f"Hasil prediksi XGBoost: **{status}**")
        a, b = st.columns(2)
        a.metric("Peluang penurunan QoS", f"{result['probability_qos_decrease'] * 100:.1f}%")
        b.metric("Kelas prediksi", status)
        st.progress(float(max(0, min(1, result["probability_qos_decrease"]))))
        if samples is not None:
            last = samples[-1]
            st.caption("Parameter QoS pada sampel terakhir")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latency", f"{last['latency_ms']:.1f} ms")
            c2.metric("Jitter", f"{last['jitter_ms']:.1f} ms")
            c3.metric("Packet loss", f"{last['packet_loss_percent']:.1f}%")
            c4.metric("Throughput", f"{last['throughput_mbps']:.2f} Mbps")


def start_page_monitoring():
    st.session_state["halaman"] = "Mulai Monitoring"


st.title("LiveStreamQoS")
st.caption("Monitoring dan prediksi kualitas layanan jaringan WiFi menggunakan XGBoost")

# Tiga menu utama sesuai kebutuhan aplikasi.
if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Home"
page = st.radio(
    "Menu navigasi", ["Home", "Mulai Monitoring", "About"],
    horizontal=True, label_visibility="collapsed", key="halaman",
)
st.divider()

if page == "Home":
    st.header("Home")
    st.subheader("Selamat datang di LiveStreamQoS")
    st.write(
        "LiveStreamQoS merupakan aplikasi untuk menguji dan memprediksi kondisi "
        "Quality of Service (QoS) jaringan WiFi dengan model Machine Learning XGBoost "
        "terbaik dari penelitian. Sistem mengolah data QoS dan menghasilkan "
        "klasifikasi **Normal** atau **Gangguan** berdasarkan prediksi model."
    )
    st.markdown("### Parameter yang dianalisis")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Latency**")
            st.caption("Waktu tunda pengiriman data, dalam milidetik (ms).")
        with st.container(border=True):
            st.markdown("**Packet Loss**")
            st.caption("Persentase paket data yang hilang saat pengiriman (%).")
    with col2:
        with st.container(border=True):
            st.markdown("**Jitter**")
            st.caption("Variasi waktu tunda antar-paket, dalam milidetik (ms).")
        with st.container(border=True):
            st.markdown("**Throughput**")
            st.caption("Jumlah data yang berhasil ditransfer per detik (Mbps).")
    st.markdown("### Cara menggunakan")
    st.write(
        "Buka **Mulai Monitoring**, lalu pilih pengujian skenario, input manual, "
        "atau unggah CSV. Aplikasi akan menganalisis 12 sampel QoS dan "
        "menampilkan hasil model XGBoost. Informasi program tersedia di menu **About**."
    )
    st.button("Mulai Monitoring", type="primary", on_click=start_page_monitoring)

elif page == "Mulai Monitoring":
    st.header("Mulai Monitoring")
    st.caption("Jenis jaringan: WiFi | Model: XGBoost | Input: 12 sampel × 4 parameter QoS")
    st.write(
        "Pilih sumber data untuk menguji hasil prediksi **Normal** atau **Gangguan**. "
        "Model XGBoost menentukan kelas dari data yang dimasukkan."
    )
    source = st.radio(
        "Sumber data", ["Uji skenario", "Input manual", "Unggah CSV"],
        horizontal=True, key="sumber_data",
    )
    st.divider()

    current = None
    if source == "Uji skenario":
        st.subheader("Pengujian Normal dan Gangguan")
        st.write("Jalankan data contoh untuk melihat hasil prediksi dari model XGBoost.")
        selected = st.selectbox(
            "Pilih skenario", ["Keduanya (Normal dan Gangguan)", "QoS baik", "QoS menurun"],
        )
        if st.button("Mulai Monitoring", type="primary", key="start_scenario"):
            try:
                scenarios = (
                    [("Skenario QoS baik", "normal"), ("Skenario QoS menurun", "gangguan")]
                    if selected == "Keduanya (Normal dan Gangguan)" else
                    [("Skenario QoS baik", "normal")] if selected == "QoS baik" else
                    [("Skenario QoS menurun", "gangguan")]
                )
                current = [(title, infer(create_scenario(scenario)), create_scenario(scenario))
                           for title, scenario in scenarios]
                st.session_state["hasil_monitoring"] = (source, current)
            except Exception as exc:
                st.session_state.pop("hasil_monitoring", None)
                st.error(f"Prediksi gagal: {exc}")
        with st.expander("Lihat data contoh 12 interval"):
            st.write("**Data QoS baik**")
            st.dataframe(to_dataframe(create_scenario("normal")), hide_index=True, use_container_width=True)
            st.write("**Data QoS menurun**")
            st.dataframe(to_dataframe(create_scenario("gangguan")), hide_index=True, use_container_width=True)

    elif source == "Input manual":
        st.subheader("Input parameter QoS")
        st.write("Ubah nilai 12 sampel berikut, dari baris paling lama hingga terbaru.")
        edited = st.data_editor(
            to_dataframe(create_scenario("normal")),
            key="tabel_qos", num_rows="fixed", hide_index=True, use_container_width=True,
            column_config={
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", min_value=0.0, format="%.2f"),
                "jitter_ms": st.column_config.NumberColumn("Jitter (ms)", min_value=0.0, format="%.2f"),
                "packet_loss_percent": st.column_config.NumberColumn("Packet Loss (%)", min_value=0.0, max_value=100.0, format="%.2f"),
                "throughput_mbps": st.column_config.NumberColumn("Throughput (Mbps)", min_value=0.001, format="%.3f"),
            },
        )
        if st.button("Mulai Monitoring", type="primary", key="start_manual"):
            try:
                rows = edited.to_dict(orient="records")
                current = [("Hasil pengujian manual", infer(rows), rows)]
                st.session_state["hasil_monitoring"] = (source, current)
            except Exception as exc:
                st.session_state.pop("hasil_monitoring", None)
                st.error(f"Data tidak valid: {exc}")

    else:
        st.subheader("Unggah data QoS dalam CSV")
        st.write("CSV harus memuat kolom latency_ms, jitter_ms, packet_loss_percent, dan throughput_mbps.")
        st.download_button(
            "Unduh contoh CSV", data=to_dataframe(create_scenario("normal")).to_csv(index=False).encode("utf-8"),
            file_name="contoh_qos_12_sampel.csv", mime="text/csv",
        )
        upload = st.file_uploader("Pilih file CSV", type=["csv"])
        rows = None
        if upload is not None:
            try:
                df = pd.read_csv(upload, nrows=10001)
                if not (WINDOW_SIZE <= len(df) <= 10000):
                    raise ValueError("CSV harus berisi minimal 12 dan maksimal 10.000 baris.")
                if any(col not in df.columns for col in SOURCE_COLUMNS):
                    raise ValueError("Kolom CSV belum lengkap.")
                df = df.loc[:, list(SOURCE_COLUMNS)].tail(WINDOW_SIZE).copy()
                for col in SOURCE_COLUMNS:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                rows = df.to_dict(orient="records")
                st.caption("Model akan menggunakan 12 baris terakhir dari CSV.")
                st.dataframe(df, hide_index=True, use_container_width=True)
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f"CSV tidak dapat digunakan: {exc}")
        if st.button("Mulai Monitoring", type="primary", disabled=rows is None, key="start_csv"):
            try:
                current = [("Hasil data CSV", infer(rows), rows)]
                st.session_state["hasil_monitoring"] = (source, current)
            except Exception as exc:
                st.session_state.pop("hasil_monitoring", None)
                st.error(f"Prediksi gagal: {exc}")

    saved = st.session_state.get("hasil_monitoring")
    if saved and saved[0] == source:
        st.markdown("### Hasil Monitoring")
        for idx, (title, result, rows) in enumerate(saved[1]):
            render_result(title, result, rows)
        st.caption("Hasil terakhir dari input yang dijalankan. Tekan Mulai Monitoring kembali setelah mengubah data.")

elif page == "About":
    st.header("About")
    st.subheader("Tentang LiveStreamQoS")
    st.write(
        "**LiveStreamQoS** merupakan aplikasi penelitian untuk menguji klasifikasi kualitas "
        "layanan jaringan WiFi menggunakan algoritma **Extreme Gradient Boosting (XGBoost)**. "
        "Aplikasi menyediakan tampilan ringkas bagi pengguna untuk memasukkan atau memilih "
        "data QoS dan melihat prediksi kondisi Normal atau Gangguan."
    )
    st.markdown("### Cara kerja program")
    st.write(
        "Empat parameter QoS yaitu latency, jitter, packet loss, dan throughput, "
        "dihimpun dalam 12 interval (masing-masing mewakili 5 detik). "
        "Aplikasi membentuk 24 fitur statistik dan mengirimkannya ke pipeline XGBoost "
        "terbaik penelitian untuk menghasilkan prediksi QoS 60 detik ke depan."
    )
    st.markdown("### Fitur utama")
    st.markdown(
        "- **Home:** pengenalan aplikasi dan parameter QoS.\n"
        "- **Mulai Monitoring:** pengujian skenario dan analisis input manual/CSV.\n"
        "- **About:** penjelasan program dan metode prediksi."
    )
    st.markdown("### Informasi penelitian")
    st.write(
        "Model menggunakan kelas **0: Normal QoS indikatif** dan "
        "**1: Penurunan QoS indikatif (Gangguan)**. Skenario bawaan memakai data sintetis; "
        "untuk menguji data jaringan nyata, pengguna dapat memasukkan pengukuran sendiri "
        "melalui input manual atau CSV. Aplikasi tidak mengukur koneksi WiFi secara otomatis."
    )
    st.caption(
        "Validasi akhir penelitian perlu memastikan rumus fitur statistik pada aplikasi "
        "sama dengan yang digunakan saat pelatihan model."
    )
