"""LiveStreamQoS: Home, Mulai Monitoring, About.

Pengukuran WiFi dilakukan oleh agen lokal di laptop pengguna. Backend Streamlit
Cloud hanya menjalankan inferensi model dari data nyata yang dikirim browser.
"""
from __future__ import annotations

from pathlib import Path
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import declare_component

st.set_page_config(page_title="LiveStreamQoS", page_icon="📶", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
BRIDGE_DIR = BASE / "component_wifi"
BRIDGE = declare_component("live_wifi_agent", path=str(BRIDGE_DIR))
QOS_COLS = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]
STATS = ["last", "mean", "min", "max", "std", "delta"]
FEATURES = [f"{stat}_{col}" for stat in STATS for col in QOS_COLS]
HISTORY_COUNT = 12


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float):
    """Gunakan model terlatih asli; tidak ada label atau ambang indikatif buatan aplikasi."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("File model tidak sesuai format bundle penelitian.")
    if bundle.get("feature_version") != "qos_24_summary_v1":
        raise ValueError("Versi fitur model tidak sesuai.")
    if list(bundle.get("features", [])) != FEATURES:
        raise ValueError("Urutan fitur model tidak sesuai.")
    if int(bundle.get("history_intervals", 0)) != HISTORY_COUNT:
        raise ValueError("Jumlah interval model tidak sesuai.")
    if int(bundle.get("interval_seconds", 0)) != 5:
        raise ValueError("Interval model bukan 5 detik.")
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Model terbaik tidak tersedia di dalam bundle.")
    # Memverifikasi metadata algoritma tanpa menampilkan namanya di antarmuka.
    if str(bundle.get("best_model_name", "")).lower() != "xgboost":
        raise ValueError("Bundle ini bukan model terbaik yang dipilih penelitian.")
    return bundle, model


def to_features(valid_rows: list[dict]) -> dict:
    if len(valid_rows) != HISTORY_COUNT:
        raise ValueError("Pengukuran belum mencapai 12 interval valid.")
    arr = pd.DataFrame(valid_rows, columns=QOS_COLS).to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Nilai pengukuran belum lengkap.")
    values = {}
    for stat in STATS:
        for i, col in enumerate(QOS_COLS):
            series = arr[:, i]
            values[f"{stat}_{col}"] = {
                "last": lambda: float(series[-1]),
                "mean": lambda: float(np.mean(series)),
                "min": lambda: float(np.min(series)),
                "max": lambda: float(np.max(series)),
                "std": lambda: float(np.std(series, ddof=0)),
                "delta": lambda: float(series[-1] - series[0]),
            }[stat]()
    return values


def predict(valid_rows: list[dict]):
    bundle, model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
    frame = pd.DataFrame([to_features(valid_rows)], columns=bundle["features"])
    proba = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    issue = float(proba[classes.index(1)])
    normal = float(proba[classes.index(0)])
    threshold = float(bundle.get("threshold", 0.5))
    return {"label": "Gangguan" if issue >= threshold else "Normal",
            "normal": normal, "gangguan": issue,
            "threshold": threshold,
            "timestamp": valid_rows[-1]["interval_end"]}


def is_valid(row: dict) -> bool:
    try:
        return (row.get("read_status") == "ok" and
                4.0 <= float(row.get("interval_seconds", 0)) <= 6.0 and
                all(np.isfinite(float(row.get(col))) for col in QOS_COLS))
    except (ValueError, TypeError):
        return False


def latest_contiguous_rows(rows: list[dict]) -> list[dict]:
    """Gunakan hanya interval valid beruntun dalam satu sesi, bukan mencampur jeda."""
    result: list[dict] = []
    prev = None
    for row in reversed(rows):
        if not is_valid(row):
            break
        try:
            t = pd.to_datetime(row["interval_end"])
            if prev is not None and (prev - t).total_seconds() > 6.8:
                break
        except Exception:
            break
        result.append(row)
        prev = t
        if len(result) >= HISTORY_COUNT:
            break
    return result[::-1]


def home():
    st.header("Selamat Datang di LiveStreamQoS")
    st.write(
        "LiveStreamQoS adalah aplikasi pemantauan kualitas layanan jaringan (Quality of Service/QoS) "
        "yang membantu pengguna mengetahui kondisi koneksi WiFi dan memperoleh prediksi "
        "**Normal** atau **Gangguan** berdasarkan hasil pengukuran jaringan."
    )
    st.subheader("Parameter yang Dipantau")
    a, b = st.columns(2)
    with a:
        st.markdown("**Latency**  \nWaktu respons koneksi menuju tujuan pengukuran, dalam milidetik (ms).")
        st.markdown("**Packet Loss**  \nPersentase paket uji tanpa respons. Pada metode cadangan TCP, ditampilkan sebagai tingkat kegagalan koneksi indikatif.")
    with b:
        st.markdown("**Jitter**  \nVariasi waktu respons koneksi, dalam milidetik (ms).")
        st.markdown("**Throughput Upload**  \nLaju data yang dikirim adaptor WiFi selama interval pemantauan, dalam Mbps.")
    st.subheader("Cara Menggunakan")
    st.write(
        "Buka menu **Mulai Monitoring**, unduh paket pengukur WiFi Windows dari tombol yang tersedia, lalu jalankan agennya di laptop yang ingin diuji. "
        "Kemudian klik **Mulai Monitoring** di halaman aplikasi. Data dikumpulkan setiap 5 detik. "
        "Hasil prediksi tersedia setelah 12 interval valid terkumpul."
    )


def about():
    st.header("About LiveStreamQoS")
    st.write(
        "LiveStreamQoS dikembangkan sebagai sistem pendukung penelitian untuk memantau parameter "
        "kualitas layanan jaringan WiFi dan mengidentifikasi indikasi penurunan QoS. "
        "Aplikasi terdiri dari halaman web yang dapat diakses secara online, agen pengukur pada laptop, "
        "dan model prediksi hasil pelatihan penelitian."
    )
    st.subheader("Cara Kerja")
    st.write(
        "Agen lokal mengambil latency, jitter, packet loss, dan throughput dari koneksi perangkat "
        "setiap 5 detik. Setelah 12 interval valid tersedia, aplikasi membentuk 24 fitur statistik "
        "dan menggunakan model penelitian untuk memperkirakan indikasi Normal atau Gangguan "
        "selama sekitar 60 detik berikutnya."
    )
    st.subheader("Penggunaan dalam Pengujian")
    st.write(
        "Responden dapat menilai kemudahan penggunaan aplikasi melalui kuesioner System Usability Scale (SUS) "
        "yang disediakan terpisah oleh peneliti. Hasil SUS menilai kegunaan antarmuka, "
        "bukan akurasi klasifikasi model."
    )
    st.subheader("Informasi Pengukuran")
    st.write(
        "Agen harus berjalan pada laptop yang WiFi-nya akan diuji. Website online tidak bisa membaca "
        "adaptor WiFi pengunjung tanpa program pendamping dan izin akses perangkat. Throughput adalah "
        "laju unggah adaptor, bukan hasil speed test. Jika perintah ICMP ping tidak tersedia, agen "
        "menggunakan koneksi TCP sebagai pengukuran indikatif dan menandainya secara eksplisit. "
        "Prediksi kualitas layanan tidak secara langsung membuktikan buffering pada video streaming."
    )


def fmt(value, digits=2):
    try:
        x = float(value)
        return f"{x:.{digits}f}" if np.isfinite(x) else "—"
    except (TypeError, ValueError):
        return "—"


@st.cache_data(show_spinner=False)
def paket_wifi_windows() -> bytes:
    """Buat paket lengkap agar berkas .bat tidak terpisah dari agen Python."""
    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as archive:
        for name in ("start_wifi.bat", "wifi_agent.py"):
            file_path = BASE / name
            if not file_path.is_file():
                raise FileNotFoundError(f"Komponen pengukur WiFi belum tersedia: {name}")
            archive.write(file_path, arcname=name)
        archive.writestr(
            "CARA_MENGGUNAKAN.txt",
            "LiveStreamQoS - Pengukur WiFi Windows\n\n"
            "1. Ekstrak semua file ZIP ini ke satu folder pada laptop Windows.\n"
            "2. Pastikan Python 3 sudah terinstal dan terhubung ke internet.\n"
            "3. Klik dua kali start_wifi.bat; jangan jalankan dari dalam ZIP.\n"
            "4. Biarkan jendela pengukur terbuka.\n"
            "5. Buka https://livestreamqos.streamlit.app dan tekan Mulai Monitoring.\n"
            "6. Izinkan akses jaringan lokal bila browser memintanya.\n"
            "7. Untuk menghentikan agen, tutup jendela terminalnya.\n\n"
            "File start_wifi.bat menjalankan wifi_agent.py di laptop ini.\n"
            "Website tidak dapat memasang atau menyalakan program lokal secara otomatis.\n"
        )
    return buf.getvalue()


def monitoring():
    st.header("Mulai Monitoring")
    st.write("Pantau kualitas WiFi laptop setiap 5 detik dan lihat prediksi setelah 12 interval valid.")
    st.markdown("**Pengukur WiFi untuk laptop Windows**")
    try:
        st.download_button(
            label="⬇️ Unduh Pengukur WiFi (Windows)",
            data=paket_wifi_windows(),
            file_name="LiveStreamQoS_WiFi.zip",
            mime="application/zip",
            help="Termasuk start_wifi.bat, wifi_agent.py, dan petunjuk penggunaan."
        )
    except (FileNotFoundError, OSError) as exc:
        st.error(f"Paket pengukur WiFi tidak tersedia: {exc}")
    st.caption("Ekstrak file ZIP di laptop Windows, kemudian klik dua kali start_wifi.bat. Python 3 harus terpasang. Unduhan tidak menjalankan agen secara otomatis.")
    with st.expander("Cara menghubungkan WiFi laptop", expanded=False):
        st.markdown(
            "1. Klik tombol **Unduh Pengukur WiFi (Windows)** di atas, lalu ekstrak ZIP.\n"
            "2. Klik dua kali **start_wifi.bat** dari folder hasil ekstraksi.\n"
            "3. Biarkan jendela agen terbuka dan pastikan laptop terhubung ke WiFi.\n"
            "4. Kembali ke halaman ini, lalu klik **Mulai Monitoring**.\n"
            "5. Jika Chrome meminta izin **akses jaringan lokal**, pilih **Izinkan**."
        )
    data = BRIDGE(key="wifi_bridge_component", default=None)
    if not isinstance(data, dict):
        st.info("Belum terhubung ke agen WiFi laptop. Jalankan start_wifi.bat, lalu klik Mulai Monitoring.")
        return

    rows = data.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    latest = rows[-1] if rows else None
    if latest:
        st.caption(f"Status: {'aktif' if data.get('active') else 'berhenti'} | "
                   f"Adaptor WiFi: {data.get('interface', '—')} | "
                   f"Pengukuran terakhir: {latest.get('interval_end', '—')}")
        cols = st.columns(4)
        for c, (label, column, unit, places) in zip(cols, [
            ("Latency", "latency_ms", "ms", 2),
            ("Jitter", "jitter_ms", "ms", 2),
            ("Packet Loss", "packet_loss_percent", "%", 1),
            ("Throughput Upload", "throughput_mbps", "Mbps", 4),
        ]):
            c.metric(label, f"{fmt(latest.get(column), places)} {unit}")
        if latest.get("note"):
            st.caption(f"Metode pengukuran: {latest.get('method', '—')}. {latest['note']}")
    else:
        st.info("Monitoring siap. Data pertama muncul setelah interval pengukuran selesai.")

    valid = latest_contiguous_rows(rows)
    st.subheader("Hasil Prediksi")
    st.progress(len(valid) / HISTORY_COUNT)
    st.caption(f"Riwayat QoS valid berurutan: {len(valid)}/{HISTORY_COUNT}")
    if len(valid) == HISTORY_COUNT:
        try:
            result = predict(valid)
            c1, c2, c3 = st.columns(3)
            c1.metric("Kondisi", result["label"])
            c2.metric("Probabilitas Normal", f"{result['normal']*100:.2f}%")
            c3.metric("Probabilitas Gangguan", f"{result['gangguan']*100:.2f}%")
            st.caption(f"Prediksi untuk sekitar 60 detik ke depan. Waktu: {result['timestamp']}")
        except Exception as exc:
            st.error(f"Prediksi belum dapat dibuat: {exc}")
    else:
        st.info("Perlu 12 interval QoS valid berurutan sebelum prediksi ditampilkan.")

    if rows:
        with st.expander("Lihat riwayat pengukuran", expanded=False):
            df = pd.DataFrame(rows)
            keep = [k for k in ["interval_end", "read_status", *QOS_COLS] if k in df.columns]
            st.dataframe(df[keep].iloc[::-1], hide_index=True, use_container_width=True)
            st.download_button("Unduh data pengukuran CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="LiveStreamQoS_pengukuran.csv", mime="text/csv")


def main():
    st.title("LiveStreamQoS")
    st.caption("Aplikasi Pemantauan Kualitas Layanan Jaringan")
    page = st.radio("Navigasi", ["Home", "Mulai Monitoring", "About"],
                    horizontal=True, label_visibility="collapsed")
    st.divider()
    if page == "Home":
        home()
    elif page == "Mulai Monitoring":
        monitoring()
    else:
        about()


if __name__ == "__main__":
    main()
