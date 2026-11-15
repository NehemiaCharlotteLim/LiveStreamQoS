"""LiveStreamQoS: demonstrasi inferensi model XGBoost terbaik tanpa pengukuran HTTP.
Buka dengan: streamlit run LiveStreamQoS_Monitoring/app.py
"""
from __future__ import annotations

import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LiveStreamQoS | XGBoost", page_icon="📶", layout="centered")
st.title("LiveStreamQoS")
st.caption("Pengujian Normal / Gangguan berdasarkan prediksi model XGBoost terbaik penelitian")
st.info("Jenis jaringan: **WiFi**. Label ini ditetapkan untuk tampilan uji, bukan hasil deteksi otomatis jaringan.")

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "models" / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"


@st.cache_resource(show_spinner=False)
def get_model(path_string: str, file_mtime_ns: int):
    # Import berat dilakukan di sini supaya error dependensi/model muncul di halaman.
    from inference import load_model_bundle
    return load_model_bundle(Path(path_string))


try:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model tidak ada di {MODEL_FILE}")
    model, bundle = get_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
    import pandas as pd
    from inference import SOURCE_COLUMNS, WINDOW_SIZE, predict_window
    from scenarios import create_scenario, to_dataframe
except Exception as exc:
    st.error("Aplikasi belum siap karena model atau dependensi gagal dimuat.")
    st.exception(exc)
    st.markdown("Periksa apakah `requirements.txt` dan `models/*.joblib` ada satu folder dengan `app.py`, lalu lihat Cloud logs.")
    st.stop()

st.success("Model XGBoost asli berhasil dimuat. Siap menguji data 12 interval × 4 parameter QoS.")


def show_result(name: str, result: dict):
    st.markdown("#### " + name)
    if result["raw_class"] == 0:
        st.success("Prediksi model: NORMAL (indikatif)")
    else:
        st.error("Prediksi model: GANGGUAN (indikatif)")
    st.caption("Label kelas penelitian: " + result["model_label"])
    c1, c2 = st.columns(2)
    c1.metric("Probabilitas penurunan QoS", f"{result['probability_qos_decrease'] * 100:.1f}%")
    c2.metric("Kelas model", str(result["raw_class"]))
    st.progress(max(0.0, min(1.0, result["probability_qos_decrease"])))
    st.caption("Prediksi mengacu pada indikasi QoS 60 detik berikutnya, bukan verifikasi gangguan pemutaran video.")


def run_model(samples):
    return predict_window(model, bundle, samples)


normal_tab, manual_tab, about_tab = st.tabs([
    "Uji Normal dan Gangguan", "Input Manual / CSV", "Info dan Diagnostik"
])

with normal_tab:
    st.subheader("Uji dua skenario menggunakan XGBoost")
    st.write("Setiap skenario menggunakan 12 sampel **sintetis** yang mewakili interval 5 detik. "
             "Kelas ditentukan dari `model.predict()`, bukan dari aturan ambang pada aplikasi.")
    if st.button("Jalankan Uji Normal dan Gangguan", type="primary", key="run_scenarios"):
        try:
            st.session_state["results_scenarios"] = (
                run_model(create_scenario("normal")),
                run_model(create_scenario("gangguan")),
            )
        except Exception as exc:
            st.session_state.pop("results_scenarios", None)
            st.error(f"Pengujian gagal: {exc}")
    if "results_scenarios" in st.session_state:
        good, bad = st.session_state["results_scenarios"]
        show_result("Skenario QoS baik", good)
        st.divider()
        show_result("Skenario QoS menurun", bad)
    with st.expander("Lihat data 12 interval pada setiap skenario"):
        st.write("Skenario QoS baik")
        st.dataframe(to_dataframe(create_scenario("normal")), width="stretch", hide_index=True)
        st.write("Skenario QoS menurun")
        st.dataframe(to_dataframe(create_scenario("gangguan")), width="stretch", hide_index=True)

with manual_tab:
    st.subheader("Uji menggunakan data Anda")
    st.write("Masukkan 12 sampel berurutan, mulai dari yang paling lama. "
             "Anda dapat mengedit tabel atau mengunggah CSV hasil pengukuran dari perangkat lain.")
    mode = st.radio("Sumber data", ["Edit tabel", "Impor CSV"], horizontal=True)
    rows = None
    if mode == "Edit tabel":
        edited = st.data_editor(
            to_dataframe(create_scenario("normal")),
            key="editor_qos", num_rows="fixed", width="stretch", hide_index=True,
            column_config={
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", min_value=0.0, format="%.2f"),
                "jitter_ms": st.column_config.NumberColumn("Jitter (ms)", min_value=0.0, format="%.2f"),
                "packet_loss_percent": st.column_config.NumberColumn("Packet Loss (%)", min_value=0.0, max_value=100.0, format="%.2f"),
                "throughput_mbps": st.column_config.NumberColumn("Throughput (Mbps)", min_value=0.001, format="%.3f"),
            },
        )
        rows = edited.to_dict(orient="records")
    else:
        st.download_button(
            "Unduh contoh CSV 12 interval",
            data=to_dataframe(create_scenario("normal")).to_csv(index=False).encode("utf-8"),
            file_name="contoh_data_qos_12_interval.csv", mime="text/csv"
        )
        uploaded = st.file_uploader("Unggah CSV dengan empat parameter QoS", type=["csv"])
        if uploaded:
            try:
                raw = pd.read_csv(uploaded, nrows=10001)
                if len(raw) < WINDOW_SIZE:
                    raise ValueError("CSV harus memiliki sedikitnya 12 baris.")
                if len(raw) > 10000:
                    raise ValueError("CSV maksimal 10.000 baris.")
                if any(c not in raw.columns for c in SOURCE_COLUMNS):
                    raise ValueError("Kolom CSV harus memuat: " + ", ".join(SOURCE_COLUMNS))
                df = raw.loc[:, list(SOURCE_COLUMNS)].tail(WINDOW_SIZE).copy()
                for column in SOURCE_COLUMNS:
                    df[column] = pd.to_numeric(df[column], errors="raise")
                rows = df.to_dict(orient="records")
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption("Model memakai 12 baris terakhir dari CSV.")
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
                st.error("CSV tidak valid: " + str(exc))
    if st.button("Prediksi dengan XGBoost", type="primary", disabled=rows is None, key="predict_manual"):
        try:
            st.session_state["manual_result"] = run_model(rows)
        except Exception as exc:
            st.session_state.pop("manual_result", None)
            st.error("Prediksi gagal: " + str(exc))
    if "manual_result" in st.session_state:
        st.caption("Hasil dari input terakhir yang diprediksi. Klik Prediksi lagi setelah data berubah.")
        show_result("Hasil data pengguna", st.session_state["manual_result"])

with about_tab:
    st.subheader("Metodologi prediksi")
    st.write("Model yang digunakan adalah **XGBoost asli** dari bundle penelitian dengan preprocessing "
             "`SimpleImputer`. Empat parameter QoS pada 12 interval dirangkum menjadi "
             "24 fitur statistik (`last`, `mean`, `min`, `max`, `std`, `delta`).")
    st.write("Label kelas model: **0 = Normal QoS indikatif**, **1 = Penurunan QoS indikatif** "
             "(ditampilkan sebagai Gangguan). Horizon prediksi model adalah 60 detik ke depan.")
    st.warning("Skenario bawaan adalah data sintetis untuk demonstrasi inferensi. "
               "Aplikasi ini **tidak mengukur kondisi WiFi secara langsung**, dan tidak membuktikan "
               "gangguan streaming. Rumus `std` (ddof=1) dan `delta` (akhir - awal) "
               "perlu diverifikasi dengan notebook pelatihan untuk pengujian ilmiah akhir.")
    st.write("**Tanpa HTTP:** Aplikasi tidak melakukan permintaan `fetch()` atau probe HTTP. "
             "Untuk menggunakan data jaringan nyata, masukkan 12 pengukuran valid melalui CSV.")
    with st.expander("Diagnostik lingkungan dan model"):
        st.write("Path model:", str(MODEL_FILE))
        st.write("Ukuran model:", MODEL_FILE.stat().st_size, "byte")
        st.write("Python:", sys.version.split()[0])
        for package in ("streamlit", "xgboost", "scikit-learn", "numpy", "pandas", "joblib", "pyarrow"):
            try:
                st.write(package + ":", version(package))
            except PackageNotFoundError:
                st.write(package + ": tidak ditemukan")
        st.write("Model terbaik:", bundle.get("best_model_name"))
        st.write("Jumlah fitur:", len(bundle.get("features", [])))
