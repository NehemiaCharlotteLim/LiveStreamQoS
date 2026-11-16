"""LiveStreamQoS: Home, Mulai Monitoring, About.

Aplikasi daring; pengukuran adaptor WiFi hanya tersedia jika agen lokal
berjalan di laptop yang sama dengan browser pengakses aplikasi.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LiveStreamQoS", page_icon="📶", layout="centered")
HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "models" / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
REQUIRED = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]


@st.cache_resource(show_spinner=False)
def get_model(path: str, mtime: int):
    # Lazy import: halaman Home tetap terbuka tanpa menunggu XGBoost dimuat.
    from inference import load_model_bundle
    return load_model_bundle(Path(path))


def infer(rows):
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("Model XGBoost belum ada di folder models/.")
    from inference import predict_window
    model, bundle = get_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
    return predict_window(model, bundle, rows)


def result_card(label: str, result: dict, rows: list[dict]):
    with st.container(border=True):
        st.markdown(f"#### {label}")
        if result["raw_class"] == 0:
            st.success("Prediksi XGBoost: **NORMAL**")
        else:
            st.error("Prediksi XGBoost: **GANGGUAN**")
        c1, c2 = st.columns(2)
        c1.metric("Peluang penurunan QoS", f"{result['probability_qos_decrease'] * 100:.1f}%")
        c2.metric("Periode prediksi", "60 detik ke depan")
        st.progress(max(0.0, min(1.0, float(result["probability_qos_decrease"]))))
        if rows:
            st.caption("Nilai parameter pada interval terakhir")
            a, b, c, d = st.columns(4)
            last = rows[-1]
            a.metric("Latency", f"{last['latency_ms']:.1f} ms")
            b.metric("Jitter", f"{last['jitter_ms']:.1f} ms")
            c.metric("Packet loss", f"{last['packet_loss_percent']:.1f}%")
            d.metric("Throughput", f"{last['throughput_mbps']:.2f} Mbps")


def go_monitoring():
    st.session_state["menu"] = "Mulai Monitoring"


def home():
    st.subheader("Selamat datang di LiveStreamQoS")
    st.write(
        "LiveStreamQoS adalah aplikasi pemantauan dan prediksi kualitas layanan "
        "(Quality of Service/QoS) jaringan **WiFi** berbasis model Machine Learning **XGBoost**. "
        "Aplikasi mengolah 12 interval data untuk memperkirakan kondisi QoS 60 detik berikutnya."
    )
    st.markdown("### Empat parameter QoS")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**Latency**")
            st.caption("Waktu tunda respons paket (ms).")
        with st.container(border=True):
            st.markdown("**Packet loss**")
            st.caption("Persentase paket percobaan yang tidak mendapat respons (%).")
    with b:
        with st.container(border=True):
            st.markdown("**Jitter**")
            st.caption("Variasi latency antar-respons (ms).")
        with st.container(border=True):
            st.markdown("**Throughput**")
            st.caption("Laju data yang teramati pada adaptor jaringan (Mbps).")
    st.markdown("### Mulai menggunakan")
    st.write(
        "Pilih **Mulai Monitoring** untuk mengambil pengukuran dari agen WiFi di laptop, "
        "atau menguji model melalui data contoh, input manual, dan CSV. "
        "Penjelasan program tersedia pada menu **About**."
    )
    st.button("Mulai Monitoring", type="primary", on_click=go_monitoring)


def monitor_automatic():
    st.markdown("#### Monitoring WiFi otomatis")
    st.write(
        "Untuk membaca jaringan WiFi laptop, jalankan `wifi_agent.py` pada laptop yang sama. "
        "Aplikasi Streamlit tetap diakses melalui internet."
    )
    with st.expander("Cara menghubungkan WiFi laptop", expanded=False):
        st.code("py -m pip install psutil\npy wifi_agent.py --origin https://livestreamqos.streamlit.app", language="powershell")
        st.caption("Ganti alamat `--origin` dengan URL Streamlit Anda jika berbeda. Buka aplikasi dari browser laptop yang menjalankan agen.")
    if "auto_run" not in st.session_state:
        st.session_state["auto_run"] = False
    a, b = st.columns(2)
    with a:
        if st.button("Mulai Monitoring", type="primary", key="start_auto", disabled=st.session_state["auto_run"]):
            st.session_state["auto_run"] = True
            st.rerun()
    with b:
        if st.button("Hentikan Monitoring", key="stop_auto", disabled=not st.session_state["auto_run"]):
            st.session_state["auto_run"] = False
            st.rerun()
    if not st.session_state["auto_run"]:
        st.info("Monitoring belum dimulai. Jalankan agen pada laptop, kemudian klik Mulai Monitoring.")
        return

    # Komponen hanya dimuat setelah monitoring diaktifkan.
    import streamlit.components.v1 as components
    component = components.declare_component("livestreamqos_local_wifi", path=str(HERE / "components" / "wifi_bridge"))
    response = component(running=True, polling_ms=5000, key="wifi_bridge", default=None)
    if not isinstance(response, dict):
        st.info("Menghubungkan browser dengan agen WiFi laptop...")
        return
    if response.get("kind") == "error":
        st.error("Agen WiFi belum dapat dihubungi. Pastikan agen berjalan di laptop yang sama dan izin akses jaringan lokal browser diberikan.")
        st.caption(str(response.get("message", ""))[:200])
        return
    if response.get("kind") != "data":
        return
    data = response.get("payload", {})
    if not isinstance(data, dict):
        st.error("Format data dari agen tidak sesuai.")
        return
    rows = data.get("samples")
    if not isinstance(rows, list):
        st.error("Format riwayat pengukuran tidak sesuai.")
        return
    n = len(rows)
    st.write(f"**Riwayat pengukuran valid: {n}/12 interval**")
    st.progress(min(1.0, n / 12))
    last_update = data.get("last_update")
    if last_update:
        st.caption(f"Pengukuran terbaru: {last_update}")
    if data.get("error"):
        st.info(str(data["error"])[:200])
    if n < 12:
        st.info("Mengumpulkan interval valid sebelum mengaktifkan prediksi XGBoost.")
        return
    try:
        result_card("Hasil monitoring WiFi", infer(rows[-12:]), rows[-12:])
    except Exception as exc:
        st.error(f"Prediksi model tidak berhasil: {exc}")
    with st.expander("Lihat 12 pengukuran terbaru"):
        import pandas as pd
        st.dataframe(pd.DataFrame(rows[-12:])[REQUIRED], hide_index=True, use_container_width=True)


def monitor_scenarios():
    from scenarios import create_scenario, to_dataframe
    st.markdown("#### Uji Normal dan Gangguan")
    st.write("Gunakan data contoh untuk menguji inferensi model XGBoost.")
    selected = st.selectbox("Skenario", ["Kedua skenario", "QoS baik", "QoS menurun"])
    if st.button("Jalankan Uji", type="primary", key="run_scenario"):
        names = (["normal", "gangguan"] if selected == "Kedua skenario" else
                 ["normal"] if selected == "QoS baik" else ["gangguan"])
        try:
            st.session_state["scenario_results"] = [
                ("Skenario QoS baik" if name == "normal" else "Skenario QoS menurun",
                 infer(create_scenario(name)), create_scenario(name)) for name in names
            ]
        except Exception as exc:
            st.session_state.pop("scenario_results", None)
            st.error(f"Gagal memprediksi skenario: {exc}")
    for label, result, rows in st.session_state.get("scenario_results", []):
        result_card(label, result, rows)
    with st.expander("Contoh data pengujian"):
        st.dataframe(to_dataframe(create_scenario("normal")), hide_index=True)
        st.dataframe(to_dataframe(create_scenario("gangguan")), hide_index=True)
    st.caption("Data skenario merupakan contoh sintetis dan tidak merepresentasikan kondisi WiFi aktual.")


def monitor_manual():
    from scenarios import create_scenario, to_dataframe
    st.markdown("#### Input manual")
    st.write("Masukkan 12 pengukuran secara berurutan, dari paling lama hingga terbaru.")
    edited = st.data_editor(
        to_dataframe(create_scenario("normal")), key="manual_table", num_rows="fixed",
        hide_index=True, use_container_width=True,
        column_config={
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", min_value=0.0),
            "jitter_ms": st.column_config.NumberColumn("Jitter (ms)", min_value=0.0),
            "packet_loss_percent": st.column_config.NumberColumn("Packet loss (%)", min_value=0.0, max_value=100.0),
            "throughput_mbps": st.column_config.NumberColumn("Throughput (Mbps)", min_value=0.0),
        },
    )
    if st.button("Prediksi Data Manual", type="primary", key="manual_run"):
        try:
            rows = edited.to_dict("records")
            st.session_state["manual_result"] = (infer(rows), rows)
        except Exception as exc:
            st.error(f"Pengujian gagal: {exc}")
            st.session_state.pop("manual_result", None)
    saved = st.session_state.get("manual_result")
    if saved:
        result_card("Hasil input manual", *saved)


def monitor_csv():
    import pandas as pd
    from scenarios import create_scenario, to_dataframe
    st.markdown("#### Unggah CSV")
    st.write("Sediakan minimal 12 baris dari empat kolom QoS.")
    st.code(",".join(REQUIRED), language="text")
    st.download_button("Unduh contoh CSV", to_dataframe(create_scenario("normal")).to_csv(index=False).encode(),
                       file_name="contoh_qos.csv", mime="text/csv")
    upload = st.file_uploader("Pilih CSV", type=["csv"], key="csv_input")
    if upload is None:
        return
    try:
        df = pd.read_csv(upload, nrows=10001)
        if len(df) < 12 or len(df) > 10000:
            raise ValueError("CSV perlu berisi 12 hingga 10.000 baris.")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError("Kolom yang tidak ada: " + ", ".join(missing))
        st.dataframe(df[REQUIRED].tail(12), hide_index=True, use_container_width=True)
        if st.button("Prediksi dari CSV", type="primary", key="csv_run"):
            rows = df[REQUIRED].tail(12).to_dict("records")
            st.session_state["csv_result"] = (infer(rows), rows)
        saved = st.session_state.get("csv_result")
        if saved:
            result_card("Hasil CSV", *saved)
    except Exception as exc:
        st.error(f"CSV belum dapat diproses: {exc}")


def about():
    st.subheader("Tentang program")
    st.write(
        "**LiveStreamQoS** dikembangkan untuk menampilkan hasil analisis Quality of Service "
        "jaringan WiFi menggunakan model **XGBoost terbaik** dari penelitian. "
        "Aplikasi dapat diakses secara online melalui Streamlit Community Cloud."
    )
    st.markdown("### Cara kerja")
    st.write(
        "Dua belas interval (masing-masing 5 detik) dari latency, jitter, packet loss, "
        "dan throughput diringkas menjadi 24 fitur: last, mean, min, max, std, dan delta. "
        "Pipeline model memprediksi kelas **Normal QoS indikatif (0)** atau "
        "**Penurunan QoS indikatif (1)** untuk 60 detik berikutnya."
    )
    st.markdown("### Sumber pengukuran")
    st.write(
        "Monitoring WiFi otomatis membutuhkan agen lokal di laptop pengguna. "
        "Agen mengukur waktu respons dan kehilangan respons ICMP ke target, "
        "serta laju transfer data adaptor WiFi. "
        "Aplikasi cloud tidak dapat membaca adaptor WiFi pengguna tanpa agen lokal. "
        "Skenario pengujian adalah data contoh, sedangkan CSV dapat berisi data pengukuran nyata."
    )
    st.write(
        "Prediksi QoS bersifat indikatif. Model ini tidak membuktikan terjadinya "
        "gangguan pada tayangan streaming tertentu. "
        "Rumus standar deviasi dan delta perlu dicocokkan dengan notebook pelatihan "
        "untuk validasi metodologi penelitian."
    )


st.title("📶 LiveStreamQoS")
st.caption("Prediksi kualitas jaringan WiFi berbasis model XGBoost")
if "menu" not in st.session_state:
    st.session_state["menu"] = "Home"
page = st.radio("Navigasi", ["Home", "Mulai Monitoring", "About"], key="menu", horizontal=True,
                label_visibility="collapsed")
st.divider()
if page == "Home":
    home()
elif page == "Mulai Monitoring":
    st.subheader("Mulai Monitoring")
    st.caption("Jenis jaringan: WiFi")
    mode = st.selectbox("Pilih sumber data", ["WiFi otomatis (laptop)", "Uji Normal dan Gangguan", "Input manual", "Unggah CSV"])
    if mode == "WiFi otomatis (laptop)":
        monitor_automatic()
    elif mode == "Uji Normal dan Gangguan":
        monitor_scenarios()
    elif mode == "Input manual":
        monitor_manual()
    else:
        monitor_csv()
else:
    about()
