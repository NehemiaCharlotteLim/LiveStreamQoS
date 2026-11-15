"""LiveStreamQoS - versi demonstrasi SUS untuk Streamlit Community Cloud.
Prediksi menggunakan model XGBoost yang telah dilatih, dengan data QoS contoh.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LiveStreamQoS", page_icon="📶", layout="centered")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
FEATURE_FIELDS = ("latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps")
STATS = ("last", "mean", "min", "max", "std", "delta")
FEATURE_NAMES = [f"{s}_{f}" for s in STATS for f in FEATURE_FIELDS]
SAMPLE_FILES = {
    "Contoh A": "contoh_qos_1.csv",
    "Contoh B": "contoh_qos_2.csv",
}


def load_sample(filename):
    if filename not in SAMPLE_FILES.values():
        raise ValueError("Contoh data tidak dikenali.")
    with (HERE / filename).open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 12:
        raise ValueError("Contoh data harus berisi tepat 12 interval.")
    samples = []
    for record in records:
        values = {name: float(record[name]) for name in FEATURE_FIELDS}
        if not all(math.isfinite(x) and x >= 0 for x in values.values()):
            raise ValueError("Terdapat pengukuran yang tidak valid.")
        if values["packet_loss_percent"] > 100:
            raise ValueError("Packet Loss tidak valid.")
        samples.append(values)
    return samples


def build_features(samples):
    if len(samples) != 12:
        raise ValueError("Diperlukan 12 interval pengukuran.")
    values = {}
    for field in FEATURE_FIELDS:
        x = [row[field] for row in samples]
        avg = sum(x) / len(x)
        items = {
            "last": x[-1], "mean": avg, "min": min(x), "max": max(x),
            "std": math.sqrt(sum((v - avg) ** 2 for v in x) / len(x)),
            "delta": x[-1] - x[0],
        }
        for stat in STATS:
            values[f"{stat}_{field}"] = float(items[stat])
    return values


@st.cache_resource(show_spinner=False)
def load_model():
    # Beban pemuatan model ditunda sampai tombol prediksi ditekan.
    import joblib
    import sklearn
    import xgboost
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("File model tidak tersedia di folder aplikasi.")
    bundle = joblib.load(MODEL_PATH)
    if list(bundle.get("features", [])) != FEATURE_NAMES:
        raise ValueError("Urutan fitur model berbeda dari aplikasi.")
    if bundle.get("best_model_name") != "XGBoost":
        raise ValueError("Model terpilih tidak sesuai.")
    if str(bundle.get("sklearn_version")) != sklearn.__version__:
        raise RuntimeError("Versi scikit-learn tidak sama dengan versi pelatihan.")
    if str(bundle.get("xgboost_version")) != xgboost.__version__:
        raise RuntimeError("Versi XGBoost tidak sama dengan versi pelatihan.")
    model = bundle["model"]
    return model, float(bundle.get("threshold", 0.5))


def run_prediction(samples):
    import pandas as pd
    model, threshold = load_model()
    features = build_features(samples)
    df = pd.DataFrame([{k: features[k] for k in FEATURE_NAMES}], columns=FEATURE_NAMES)
    probabilities = model.predict_proba(df)[0]
    categories = [int(x) for x in model.classes_]
    p_bad = float(probabilities[categories.index(1)])
    return ("GANGGUAN" if p_bad >= threshold else "NORMAL"), p_bad


st.title("LiveStreamQoS")
st.caption("Pemantauan dan prediksi indikasi kondisi Quality of Service jaringan")
page_home, page_predict, page_about = st.tabs(["Home", "Prediksi QoS", "About"])

with page_home:
    st.subheader("Selamat Datang")
    st.write("LiveStreamQoS menganalisis riwayat **Latency, Jitter, Packet Loss, dan Throughput** "
             "untuk memprediksi indikasi kualitas jaringan pada 60 detik berikutnya.")
    st.write("Untuk mencoba aplikasi, buka **Prediksi QoS**, pilih data contoh, "
             "lalu tekan **Mulai Pengujian**. Setelah melihat hasil, buka halaman About.")
    st.info("Versi website ini menggunakan data QoS contoh untuk pengujian kemudahan "
            "penggunaan. Website tidak mengukur Wi-Fi pengunjung secara langsung.")

with page_predict:
    st.subheader("Prediksi QoS")
    st.write("Pilih salah satu rekaman pengukuran QoS dan jalankan prediksi menggunakan model terlatih.")
    example_name = st.selectbox("Data pengujian", list(SAMPLE_FILES))
    if st.button("Mulai Pengujian", type="primary", use_container_width=True):
        try:
            with st.spinner("Memproses data QoS..."):
                sample = load_sample(SAMPLE_FILES[example_name])
                result, p_bad = run_prediction(sample)
            st.session_state["last_prediction"] = (example_name, result, p_bad, sample[-1])
        except Exception as exc:
            st.session_state.pop("last_prediction", None)
            st.error(f"Prediksi gagal dijalankan: {exc}")

    previous = st.session_state.get("last_prediction")
    if previous and previous[0] == example_name:
        _, result, p_bad, latest = previous
        st.subheader("Hasil Prediksi")
        if result == "NORMAL":
            st.success("Status prediksi: NORMAL")
        else:
            st.warning("Status prediksi: GANGGUAN")
        left, right = st.columns(2)
        left.metric("Probabilitas Normal", f"{(1 - p_bad) * 100:.2f}%")
        right.metric("Probabilitas Gangguan", f"{p_bad * 100:.2f}%")
        st.progress(max(0.0, min(1.0, p_bad)))
        st.markdown("**Pengukuran pada interval terakhir:**")
        cols = st.columns(2)
        cols[0].metric("Latency", f"{latest['latency_ms']:.2f} ms")
        cols[1].metric("Jitter", f"{latest['jitter_ms']:.2f} ms")
        cols = st.columns(2)
        cols[0].metric("Packet Loss", f"{latest['packet_loss_percent']:.2f}%")
        cols[1].metric("Throughput", f"{latest['throughput_mbps']:.3f} Mbps")
        st.caption("Hasil dihitung dari 12 interval QoS contoh; bukan hasil pengukuran Wi-Fi pribadi.")
    else:
        st.info("Tekan Mulai Pengujian untuk melihat hasil Normal atau Gangguan.")

with page_about:
    st.subheader("Tentang Aplikasi")
    st.write("LiveStreamQoS merupakan aplikasi penelitian yang mengolah empat parameter QoS "
             "dan menampilkan kelas indikatif **Normal** atau **Gangguan** berikut probabilitasnya.")
    st.write("Model prediksi menganalisis ringkasan 12 interval pengukuran (sekitar 60 detik) "
             "sebagai dasar perkiraan kondisi 60 detik berikutnya.")
    st.write("Website ini digunakan untuk demonstrasi fungsi prediksi dan pengujian kemudahan "
             "penggunaan menggunakan metode System Usability Scale (SUS). "
             "Kuesioner SUS disediakan secara terpisah oleh peneliti.")
    st.caption("Hasil hanya merupakan indikasi QoS berbasis data contoh dan bukan kepastian gangguan tayangan streaming.")
