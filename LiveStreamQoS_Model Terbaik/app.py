"""LiveStreamQoS: demonstrasi prediksi untuk pengujian kegunaan SUS.

Versi cloud menggunakan dua cuplikan data QoS pengukuran historis.
Pengunjung tidak perlu memasang sensor ataupun mengunggah berkas.
Tidak melakukan pengukuran Wi-Fi pengunjung dan bukan hasil pengujian gangguan video aktual.
"""

from __future__ import annotations
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import xgboost

st.set_page_config(
    page_title="LiveStreamQoS | Prediksi QoS",
    page_icon="📶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
BUNDLE_PATH = BASE / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
QOS = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]
STATS = ["last", "mean", "min", "max", "std", "delta"]
FEATURES = [f"{s}_{q}" for s in STATS for q in QOS]
DISPLAY = {
    "latency_ms": "Latency (ms)",
    "jitter_ms": "Jitter (ms)",
    "packet_loss_percent": "Packet Loss (%)",
    "throughput_mbps": "Throughput (Mbps)",
}
EXAMPLES = {
    "Contoh 1": "contoh_qos_1.csv",
    "Contoh 2": "contoh_qos_2.csv",
}

st.markdown(
    """<style>
      .block-container { padding-top: 2.15rem; padding-bottom: 3rem; }
      .small-muted { color: #718096; font-size: .9rem; }
      [data-testid="stMetricValue"] { font-size: 1.65rem; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Muat hanya artefak lokal yang terpercaya; verifikasi kontrak 24 fitur."""
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError("File model tidak ditemukan di folder aplikasi.")
    bundle = joblib.load(BUNDLE_PATH)
    if not isinstance(bundle, dict):
        raise ValueError("Format berkas model tidak didukung.")
    if bundle.get("feature_version") != "qos_24_summary_v1":
        raise ValueError("Versi fitur model tidak sesuai.")
    if list(bundle.get("features", [])) != FEATURES:
        raise ValueError("Susunan 24 fitur model tidak sesuai.")
    if int(bundle.get("history_intervals", 0)) != 12:
        raise ValueError("Panjang riwayat model tidak sesuai.")
    if bundle.get("best_model_name") != "XGBoost":
        raise ValueError("Berkas model terpilih tidak sesuai.")
    if str(bundle.get("sklearn_version")) != sklearn.__version__:
        raise ValueError("Versi scikit-learn tidak cocok dengan artefak model.")
    if str(bundle.get("xgboost_version")) != xgboost.__version__:
        raise ValueError("Versi pustaka model tidak cocok dengan artefak model.")
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Model tidak memiliki fungsi prediksi probabilitas.")
    if set(int(v) for v in model.classes_) != {0, 1}:
        raise ValueError("Kelas model tidak sesuai dengan Normal/Gangguan.")
    return bundle, model


@st.cache_data(show_spinner=False)
def load_example(filename: str) -> pd.DataFrame:
    if filename not in EXAMPLES.values():
        raise ValueError("Pilihan data tidak dikenal.")
    file = BASE / filename
    if not file.is_file():
        raise FileNotFoundError("Berkas data contoh tidak ditemukan.")
    data = pd.read_csv(file)
    if len(data) != 12 or any(col not in data for col in QOS):
        raise ValueError("Contoh data harus memiliki 12 baris dan empat parameter QoS.")
    data = data[QOS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    vals = data.to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("Contoh data memiliki nilai QoS tidak valid.")
    if data["packet_loss_percent"].gt(100).any():
        raise ValueError("Nilai Packet Loss di luar rentang 0 sampai 100%.")
    return data


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    matrix = data[QOS].to_numpy(dtype=float)
    features = {}
    for stat in STATS:
        for j, col in enumerate(QOS):
            v = matrix[:, j]
            if stat == "last":
                value = v[-1]
            elif stat == "mean":
                value = v.mean()
            elif stat == "min":
                value = v.min()
            elif stat == "max":
                value = v.max()
            elif stat == "std":
                value = v.std(ddof=0)
            else:
                value = v[-1] - v[0]
            features[f"{stat}_{col}"] = float(value)
    return pd.DataFrame([features], columns=FEATURES)


def predict(data: pd.DataFrame):
    bundle, model = load_model()
    probabilities = model.predict_proba(make_features(data))[0]
    classes = [int(c) for c in model.classes_]
    p_bad = float(probabilities[classes.index(1)])
    p_good = float(probabilities[classes.index(0)])
    threshold = float(bundle.get("threshold", 0.5))
    label = "Gangguan" if p_bad >= threshold else "Normal"
    return label, p_bad, p_good


def home_page():
    st.subheader("Selamat Datang")
    st.write(
        "LiveStreamQoS adalah aplikasi untuk memahami kualitas jaringan melalui "
        "**Latency, Jitter, Packet Loss, dan Throughput**. Sistem menganalisis "
        "riwayat pengukuran untuk menghasilkan prediksi indikatif **Normal** atau "
        "**Gangguan** pada periode 60 detik berikutnya."
    )
    st.markdown("#### Cara menggunakan versi website")
    st.markdown(
        "1. Buka halaman **Prediksi QoS**.\n"
        "2. Tekan **Mulai Pengujian** untuk mencoba data contoh yang tersedia.\n"
        "3. Baca hasil **Normal/Gangguan** dan probabilitasnya.\n"
        "4. Coba **Contoh 2** jika ingin melihat kondisi data berbeda.\n"
        "5. Buka **About**, lalu isi kuesioner SUS melalui tautan dari peneliti."
    )
    st.info(
        "Versi website untuk pengujian kemudahan penggunaan (SUS) memakai data "
        "QoS contoh dari pengukuran sebelumnya, bukan mengukur Wi-Fi pengunjung."
    )


def prediction_page():
    st.subheader("Prediksi QoS")
    st.write("Tekan tombol di bawah untuk mencoba fitur prediksi menggunakan 12 pengukuran QoS contoh.")
    choice = st.selectbox("Pilih data contoh", list(EXAMPLES), key="example_choice")
    filename = EXAMPLES[choice]
    try:
        data = load_example(filename)
    except Exception as exc:
        st.error(f"Data contoh tidak dapat dibaca: {exc}")
        return

    if st.button("Mulai Pengujian", type="primary", use_container_width=True):
        try:
            label, p_bad, p_good = predict(data)
            st.session_state["prediksi_terakhir"] = {
                "case": choice, "label": label,
                "p_bad": p_bad, "p_good": p_good,
            }
        except Exception as exc:
            st.session_state.pop("prediksi_terakhir", None)
            st.error(f"Prediksi belum dapat dijalankan: {exc}")
            return

    previous = st.session_state.get("prediksi_terakhir")
    if previous is not None and previous["case"] == choice:
        st.markdown("#### Hasil Prediksi")
        st.metric("Status QoS", previous["label"] .upper())
        col_a, col_b = st.columns(2)
        col_a.metric("Probabilitas Normal", f"{previous['p_good']*100:.2f}%")
        col_b.metric("Probabilitas Gangguan", f"{previous['p_bad']*100:.2f}%")
        st.progress(max(0.0, min(1.0, previous["p_bad"])))
        st.caption(
            "Prediksi merupakan indikasi keadaan QoS pada 60 detik berikutnya. "
            "Ini bukan konfirmasi bahwa video streaming benar-benar buffering."
        )
        st.markdown("#### Data QoS pada Pengukuran Terakhir")
        current = data.iloc[-1]
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        for container, field, unit, precision in (
            (k1,"latency_ms","ms",2), (k2,"jitter_ms","ms",2),
            (k3,"packet_loss_percent","%",2), (k4,"throughput_mbps","Mbps",3),
        ):
            container.metric(DISPLAY[field], f"{current[field]:.{precision}f} {unit}")
    else:
        st.caption("Hasil akan muncul setelah Anda menekan **Mulai Pengujian**.")

    with st.expander("Lihat 12 data pengukuran yang digunakan"):
        table = data.rename(columns=DISPLAY).copy()
        table.insert(0, "Interval", range(1, len(table) + 1))
        st.dataframe(table, hide_index=True, use_container_width=True)
    st.info("Sumber data: cuplikan dataset penelitian. Tidak ada pengukuran Wi-Fi pengunjung pada versi web ini.")


def about_page():
    st.subheader("Tentang LiveStreamQoS")
    st.write(
        "LiveStreamQoS dikembangkan sebagai aplikasi penelitian untuk menampilkan "
        "indikasi kualitas jaringan berdasarkan hasil analisis empat parameter QoS."
    )
    st.markdown("#### Kegunaan")
    st.markdown(
        "- Membantu pengguna memahami **Latency, Jitter, Packet Loss, dan Throughput**.\n"
        "- Memperlihatkan cara sistem memproses data jaringan menjadi prediksi.\n"
        "- Menampilkan kelas **Normal/Gangguan** serta probabilitasnya."
    )
    st.markdown("#### Cara kerja prediksi")
    st.write(
        "Dua belas interval pengukuran, masing-masing sekitar lima detik, diringkas "
        "menjadi 24 fitur statistik. Model pembelajaran mesin hasil evaluasi penelitian "
        "kemudian menghitung probabilitas kondisi QoS untuk horizon 60 detik."
    )
    st.markdown("#### Batasan penggunaan")
    st.write(
        "Pada versi publik, pengunjung mencoba data QoS yang telah tersedia. "
        "Model tidak memantau Wi-Fi pengunjung secara langsung dan keluaran "
        "bukan bukti gangguan tayangan streaming yang diamati."
    )
    st.markdown("#### Pengujian pengguna")
    st.write(
        "Responden dapat menilai pengalaman menggunakan Home, Prediksi QoS, dan About "
        "melalui kuesioner System Usability Scale (SUS) yang disediakan peneliti secara terpisah."
    )


def main():
    st.title("LiveStreamQoS")
    st.caption("Aplikasi prediksi indikasi kualitas jaringan")
    home, prediction, about = st.tabs(["Home", "Prediksi QoS", "About"])
    with home:
        home_page()
    with prediction:
        prediction_page()
    with about:
        about_page()


if __name__ == "__main__":
    main()
