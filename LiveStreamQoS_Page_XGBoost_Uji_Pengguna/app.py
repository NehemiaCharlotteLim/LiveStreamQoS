"""LiveStreamQoS | Beranda, monitoring dan prediksi QoS, serta informasi aplikasi.

Jalankan: python -m streamlit run app.py
Memakai model terbaik yang dipilih melalui validasi silang.
Tidak memodifikasi aplikasi pembanding tiga model sebelumnya.
"""

from __future__ import annotations

import csv
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import streamlit as st
import xgboost


st.set_page_config(page_title="LiveStreamQoS | Monitoring dan Prediksi QoS", page_icon="📶", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
HISTORY_DIR = APP_DIR / "riwayat_model_terbaik"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
QOS_COLS = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]
STATS = ["last", "mean", "min", "max", "std", "delta"]
FEATURES = [f"{s}_{c}" for s in STATS for c in QOS_COLS]
HISTORY_COUNT = 12
PERIOD_SECONDS = 5
PING_TARGET = "1.1.1.1"  # Target pengukuran internal, tidak ditampilkan di antarmuka.
PING_COUNT = 4
PING_WAIT_MS = 900
PING_MAX_SECONDS = 4.65

SAMPLE_COLUMNS = ["session_id", "scenario", "interval_start", "interval_end", "interval_seconds",
                  "interface_name", "target", "read_status", "latency_ms", "jitter_ms",
                  "packet_loss_percent", "throughput_mbps", "ping_error"]
PRED_COLUMNS = ["session_id", "timestamp", "model_name", "history_start", "history_end",
                "horizon_start", "horizon_end", "probabilitas_normal", "probabilitas_gangguan",
                "prediksi", "threshold", "span_history_seconds", "catatan"]

def read_model(path: Path):
    """Hanya muat berkas joblib dari sumber yang dipercaya."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("File model bukan bundle dictionary dari notebook CV.")
    if bundle.get("feature_version") != "qos_24_summary_v1":
        raise ValueError("Versi fitur model tidak sesuai (perlu qos_24_summary_v1).")
    if list(bundle.get("features", [])) != FEATURES:
        raise ValueError("Urutan 24 fitur di bundle tidak sesuai dengan aplikasi.")
    if int(bundle.get("history_intervals", 0)) != HISTORY_COUNT:
        raise ValueError("Model tidak menggunakan 12 interval historis.")
    if int(bundle.get("horizon_intervals", 0)) != 12:
        raise ValueError("Model tidak menggunakan horizon 12 interval mendatang.")
    if int(bundle.get("interval_seconds", 0)) != PERIOD_SECONDS:
        raise ValueError("Periode sampel model bukan 5 detik.")
    model_name = bundle.get("best_model_name")
    model = bundle.get("model")
    if not model_name or model is None:
        raise ValueError("Informasi model terbaik tidak ditemukan; gunakan bundle dari Colab revisi.")
    if "models" in bundle and model_name not in bundle["models"]:
        raise ValueError("Model terpilih tidak terdapat pada daftar model hasil CV.")
    train_sklearn = str(bundle.get("sklearn_version", ""))
    train_xgb = str(bundle.get("xgboost_version", ""))
    if train_sklearn and train_sklearn != sklearn.__version__:
        raise ValueError(f"Versi scikit-learn berbeda: model {train_sklearn}, aplikasi {sklearn.__version__}. Samakan versi di requirements.txt.")
    if train_xgb and train_xgb != xgboost.__version__:
        raise ValueError(f"Versi library model berbeda: model {train_xgb}, aplikasi {xgboost.__version__}. Samakan versi di requirements.txt.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model tidak mendukung predict_proba.")
    if 1 not in list(model.classes_):
        raise ValueError("Model tidak mempunyai kelas Gangguan (1).")
    return bundle, str(model_name), model


@st.cache_resource(show_spinner=False)
def cached_model(path_str: str, stamp: float):
    return read_model(Path(path_str))


def qos_features(rows: list[dict]) -> dict:
    """Sama persis dengan history_to_features pada notebook Google Colab revisi."""
    if len(rows) != HISTORY_COUNT:
        raise ValueError(f"Diperlukan {HISTORY_COUNT} sampel valid, tersedia {len(rows)}.")
    data = pd.DataFrame(rows, columns=QOS_COLS).to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Fitur memiliki nilai yang tidak terbaca.")
    result = {}
    for stat in STATS:
        for idx, col in enumerate(QOS_COLS):
            values = data[:, idx]
            result[f"{stat}_{col}"] = float({
                "last": lambda: values[-1],
                "mean": lambda: np.mean(values),
                "min": lambda: np.min(values),
                "max": lambda: np.max(values),
                "std": lambda: np.std(values, ddof=0),
                "delta": lambda: values[-1] - values[0],
            }[stat]())
    return result


def make_prediction(rows: list[dict], bundle: dict, name: str, model) -> dict:
    frame = pd.DataFrame([qos_features(rows)], columns=bundle["features"])
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    issue = float(probabilities[classes.index(1)])
    normal = float(probabilities[classes.index(0)]) if 0 in classes else 1.0 - issue
    threshold = float(bundle.get("threshold", 0.5))
    hist_start = pd.to_datetime(rows[0]["interval_start"])
    hist_end = pd.to_datetime(rows[-1]["interval_end"])
    span_seconds = (hist_end - hist_start).total_seconds()
    return {
        "session_id": rows[-1]["session_id"],
        "timestamp": rows[-1]["interval_end"],
        "model_name": name,
        "history_start": str(hist_start),
        "history_end": str(hist_end),
        "horizon_start": str(hist_end),
        "horizon_end": str(hist_end + timedelta(seconds=60)),
        "probabilitas_normal": normal,
        "probabilitas_gangguan": issue,
        "prediksi": "Gangguan" if issue >= threshold else "Normal",
        "threshold": threshold,
        "span_history_seconds": span_seconds,
        "catatan": ("Sampel valid terputus; rentang historis lebih dari 75 detik. "
                    "Interpretasi 60 detik perlu kehati-hatian." if span_seconds > 75 else ""),
    }


def _ping_command(host: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or host.startswith("-"):
        raise ValueError("Alamat ping harus berupa hostname/IP sederhana, contohnya 1.1.1.1.")
    if sys.platform.startswith("win"):
        return ["ping", "-n", str(PING_COUNT), "-w", str(PING_WAIT_MS), host]
    if sys.platform == "darwin":
        return ["ping", "-c", str(PING_COUNT), "-W", "1", host]
    return ["ping", "-c", str(PING_COUNT), "-W", "1", host]


def _ping_once(host: str) -> dict:
    try:
        completed = subprocess.run(_ping_command(host), capture_output=True, text=True,
                                   errors="replace", timeout=PING_MAX_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"latency_ms": np.nan, "jitter_ms": np.nan, "packet_loss_percent": np.nan,
                "ping_error": f"Ping gagal atau timeout: {exc}"}
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    rtts = [float(x.replace(",", ".")) for x in re.findall(
        r"(?:time|waktu)\s*[=<]\s*(\d+(?:[.,]\d+)?)\s*(?:msec|ms|md)\b", output, flags=re.I)]
    win_summary = re.search(
        r"(?:Sent|Dikirim|Terkirim)\s*=\s*(\d+)\s*,\s*"
        r"(?:Received|Diterima)\s*=\s*(\d+)\s*,\s*"
        r"(?:Lost|Hilang)\s*=\s*(\d+)", output, flags=re.I | re.S)
    unix_summary = re.search(r"(\d+)\s+packets? transmitted\s*,\s*(\d+)\s+(?:packets? )?received", output, flags=re.I)
    if win_summary:
        sent, received = int(win_summary[1]), int(win_summary[2])
    elif unix_summary:
        sent, received = int(unix_summary[1]), int(unix_summary[2])
    elif rtts:
        sent, received = PING_COUNT, len(rtts)
    else:
        sent, received = PING_COUNT, 0
    loss = float(100 * max(0, sent - received) / sent) if sent else np.nan
    latency = float(np.mean(rtts)) if rtts else np.nan
    jitter = float(np.mean(np.abs(np.diff(rtts)))) if len(rtts) >= 2 else np.nan
    if received == 0:
        note = "Tidak ada balasan ping (ICMP diblokir atau koneksi sedang gagal)."
    elif len(rtts) < 2:
        note = "Balasan ping kurang dari 2, sehingga Jitter tidak dapat dihitung."
    else:
        note = ""
    return {"latency_ms": latency, "jitter_ms": jitter,
            "packet_loss_percent": loss, "ping_error": note}


def _nic_candidates() -> list[str]:
    stats = psutil.net_if_stats()
    candidates = [name for name, info in stats.items() if info.isup and not any(
        banned in name.lower() for banned in ("loopback", "pseudo", "virtualbox", "bluetooth", "vmware"))]
    if not candidates:
        candidates = list(psutil.net_io_counters(pernic=True).keys())
    return sorted(candidates, key=lambda x: (not any(s in x.lower() for s in ("wi-fi", "wifi", "wlan", "wireless")), x.lower()))


def _append_csv(filename: str, columns: list[str], record: dict) -> None:
    path = HISTORY_DIR / filename
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)


class NetworkSampler:
    """Pengukuran di thread terpisah. Tidak memanggil fungsi Streamlit dari thread."""

    def __init__(self, nic: str, host: str, session_id: str):
        self.nic, self.host, self.session_id = nic, host, session_id
        self.q: queue.Queue[dict] = queue.Queue(maxsize=400)
        self.errors: queue.Queue[str] = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="LStreamQoS-Best", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def _loop(self):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                while not self.stop_event.is_set():
                    begin_mon = time.monotonic()
                    begin_wall = datetime.now()
                    initial = psutil.net_io_counters(pernic=True).get(self.nic)
                    job = pool.submit(_ping_once, self.host)
                    if self.stop_event.wait(PERIOD_SECONDS):
                        break
                    finish_mon = time.monotonic()
                    finish_wall = datetime.now()
                    ending = psutil.net_io_counters(pernic=True).get(self.nic)
                    duration = finish_mon - begin_mon
                    try:
                        ping = job.result(timeout=0.15)
                    except FutureTimeout:
                        ping = {"latency_ms": np.nan, "jitter_ms": np.nan,
                                "packet_loss_percent": np.nan, "ping_error": "Ping belum selesai dalam interval."}
                    except Exception as exc:
                        ping = {"latency_ms": np.nan, "jitter_ms": np.nan,
                                "packet_loss_percent": np.nan, "ping_error": str(exc)}
                    if initial is not None and ending is not None and ending.bytes_sent >= initial.bytes_sent:
                        throughput = (ending.bytes_sent - initial.bytes_sent) * 8 / duration / 1_000_000
                    else:
                        throughput = np.nan
                    nums = [ping["latency_ms"], ping["jitter_ms"], ping["packet_loss_percent"], throughput]
                    if not 4 <= duration <= 6:
                        status, note = "partial", "Jeda interval tidak mendekati 5 detik."
                    elif not all(np.isfinite(v) for v in nums):
                        status, note = "failed", ping.get("ping_error") or "Parameter QoS tidak lengkap."
                    else:
                        status, note = "ok", ""
                    result = {
                        "session_id": self.session_id, "scenario": "monitoring_langsung",
                        "interval_start": begin_wall.isoformat(sep=" ", timespec="milliseconds"),
                        "interval_end": finish_wall.isoformat(sep=" ", timespec="milliseconds"),
                        "interval_seconds": round(duration, 3),
                        "interface_name": self.nic, "target": self.host,
                        "read_status": status, **{k: ping[k] for k in QOS_COLS if k in ping},
                        "throughput_mbps": throughput, "ping_error": note,
                    }
                    try:
                        self.q.put_nowait(result)
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                            self.q.put_nowait(result)
                        except queue.Empty:
                            pass
        except Exception as exc:
            try:
                self.errors.put_nowait(f"Sampler berhenti: {exc}")
            except queue.Full:
                pass


def initialize_state():
    for key, value in {
        "best_sampler": None, "best_session_id": None,
        "best_samples": [], "best_buffer": [], "best_prediction": None,
        "best_predictions": [], "best_last_note": "", "best_sampling_error": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def drain_sampler(bundle, name: str, model):
    sampler = st.session_state.best_sampler
    if sampler is None:
        return
    for _ in range(10):
        try:
            st.session_state.best_sampling_error = sampler.errors.get_nowait()
        except queue.Empty:
            break
    for _ in range(100):
        try:
            row = sampler.q.get_nowait()
        except queue.Empty:
            break
        st.session_state.best_samples = (st.session_state.best_samples + [row])[-600:]
        _append_csv("pengukuran_qos.csv", SAMPLE_COLUMNS, row)
        values = np.asarray([row.get(col, np.nan) for col in QOS_COLS], dtype=float)
        valid = (row.get("read_status") == "ok" and np.isfinite(values).all()
                 and 4 <= float(row["interval_seconds"]) <= 6)
        if not valid:
            st.session_state.best_last_note = ("Sampel gagal tetap tercatat di CSV, "
                                               "tetapi tidak dimasukkan ke 12 sampel prediksi.")
            continue
        buffer = st.session_state.best_buffer
        if buffer and (buffer[-1]["session_id"] != row["session_id"] or
                       buffer[-1]["interface_name"] != row["interface_name"]):
            buffer = []
        buffer = (buffer + [row])[-HISTORY_COUNT:]
        st.session_state.best_buffer = buffer
        st.session_state.best_last_note = "Riwayat valid bertambah; sampel gagal tidak mengurangi progres."
        if len(buffer) == HISTORY_COUNT:
            prediction = make_prediction(buffer, bundle, name, model)
            st.session_state.best_prediction = prediction
            st.session_state.best_predictions = (st.session_state.best_predictions + [prediction])[-600:]
            _append_csv("prediksi_model_terbaik.csv", PRED_COLUMNS, prediction)


def text_number(x, places=2):
    try:
        x = float(x)
        return f"{x:,.{places}f}" if np.isfinite(x) else "—"
    except (TypeError, ValueError):
        return "—"


@st.fragment(run_every="1s")
def live_monitor_panel(bundle, model_name: str, model):
    drain_sampler(bundle, model_name, model)
    samples = st.session_state.best_samples
    if not samples:
        st.info("Belum ada hasil interval. Tekan **Mulai Monitoring**; pembacaan pertama selesai setelah sekitar 5 detik.")
        if st.session_state.best_sampling_error:
            st.error(st.session_state.best_sampling_error)
        return
    latest = samples[-1]
    cols = st.columns(4)
    for box, column, label, unit, dp in zip(cols, QOS_COLS,
         ["Latency", "Jitter", "Packet Loss", "Throughput Upload"], ["ms", "ms", "%", "Mbps"], [2, 2, 1, 4]):
        box.metric(label, f"{text_number(latest.get(column), dp)} {unit}")
    st.caption(f"Pengukuran terakhir: {latest['interval_end']} | Status: {latest['read_status']} | "
               f"Interface: {latest['interface_name']}")
    if latest.get("ping_error"):
        st.warning(f"Catatan pengukuran: {latest['ping_error']}")
    if st.session_state.best_sampling_error:
        st.error(st.session_state.best_sampling_error)

    n_valid = len(st.session_state.best_buffer)
    st.subheader("Hasil Prediksi QoS")
    st.progress(n_valid / HISTORY_COUNT)
    st.caption(f"Riwayat QoS valid: **{n_valid}/{HISTORY_COUNT}** | "
               f"{st.session_state.best_last_note}")
    prediction = st.session_state.best_prediction
    if prediction is None:
        st.info("Sistem memerlukan 12 interval QoS valid (sekitar 60 detik jika tidak ada pembacaan gagal). Hasil muncul otomatis setelah riwayat lengkap.")
    else:
        a, b, c = st.columns([1.2, 1, 1])
        a.metric("Prediksi indikasi QoS 60 detik ke depan", prediction["prediksi"])
        b.metric("Probabilitas Gangguan", f"{prediction['probabilitas_gangguan'] * 100:.2f}%")
        c.metric("Probabilitas Normal", f"{prediction['probabilitas_normal'] * 100:.2f}%")
        st.progress(min(1.0, max(0.0, prediction["probabilitas_gangguan"])))
        st.caption(f"Ambang klasifikasi: {prediction['threshold']:.2f} | "
                   f"Prediksi dibuat: {prediction['timestamp']}")
        st.caption(f"Riwayat: {prediction['history_start']} sampai {prediction['history_end']} | "
                   f"Horizon indikatif: {prediction['horizon_start']} sampai {prediction['horizon_end']}")
        if prediction["catatan"]:
            st.warning(prediction["catatan"])
        elif n_valid < HISTORY_COUNT:
            st.warning("Prediksi yang tampil merupakan prediksi terakhir yang tersimpan, bukan pengukuran baru.")
    st.caption("Output adalah prediksi **indikasi QoS**, bukan bukti bahwa tayangan video benar-benar buffering. "
               "Throughput mengukur lalu lintas **upload aktual laptop**, bukan kapasitas internet dari speed test.")
    with st.expander("Riwayat pengukuran sesi ini", expanded=False):
        cols_display = ["interval_end", "read_status"] + QOS_COLS
        data = pd.DataFrame(samples).iloc[::-1]
        st.dataframe(data[cols_display].head(80), use_container_width=True, hide_index=True)



def show_home():
    st.subheader("Selamat Datang di LiveStreamQoS")
    st.write(
        "**LiveStreamQoS** adalah aplikasi pemantauan kualitas jaringan Wi-Fi yang "
        "menampilkan parameter Quality of Service (QoS) dan perkiraan kondisi QoS "
        "untuk **60 detik berikutnya**. Aplikasi dibuat untuk membantu pengguna "
        "memahami kondisi koneksi saat melakukan aktivitas seperti live streaming."
    )
    st.markdown("#### Informasi yang Ditampilkan")
    a, b = st.columns(2)
    with a:
        st.markdown("**Latency (ms)**")
        st.write("Rata-rata waktu perjalanan pulang-pergi paket uji jaringan (RTT).")
        st.markdown("**Jitter (ms)**")
        st.write("Variasi waktu respons antarbalasan ping yang diterima.")
    with b:
        st.markdown("**Packet Loss (%)**")
        st.write("Persentase paket ping yang tidak menerima balasan.")
        st.markdown("**Throughput Upload (Mbps)**")
        st.write("Laju data keluar melalui adaptor jaringan laptop.")
    st.markdown("#### Cara Menggunakan Aplikasi")
    st.markdown(
        "1. Buka tab **Prediksi QoS**.\n"
        "2. Pilih adaptor Wi-Fi yang sedang dipakai.\n"
        "3. Klik **Mulai Monitoring**.\n"
        "4. Tunggu hingga **12 sampel valid** terkumpul.\n"
        "5. Amati nilai QoS, probabilitas, dan hasil prediksi **Normal/Gangguan**."
    )
    st.info(
        "Pengukuran berlangsung sekitar setiap 5 detik. Prediksi baru tersedia "
        "setelah riwayat QoS mencukupi. Pembacaan yang gagal tidak dihitung sebagai sampel valid."
    )
    st.caption(
        "Hasil prediksi merupakan indikasi berdasarkan parameter jaringan, "
        "bukan jaminan bahwa tayangan live streaming akan mengalami buffering."
    )


def show_about():
    st.subheader("Tentang LiveStreamQoS")
    st.write(
        "LiveStreamQoS dikembangkan sebagai aplikasi penelitian untuk mengukur "
        "kondisi jaringan Wi-Fi dan memberikan prediksi indikasi kualitas jaringan "
        "berdasarkan riwayat pengukuran."
    )
    st.markdown("#### Tujuan dan Kegunaan")
    st.markdown(
        "- Membantu pengguna memantau perubahan kualitas koneksi Wi-Fi.\n"
        "- Menampilkan Latency, Jitter, Packet Loss, dan Throughput dalam satu halaman.\n"
        "- Memberikan informasi indikatif **Normal** atau **Gangguan** untuk 60 detik berikutnya.\n"
        "- Mendukung pengamatan dan evaluasi kondisi jaringan saat live streaming."
    )
    st.markdown("#### Cara Kerja")
    st.write(
        "Aplikasi mencatat empat parameter QoS setiap 5 detik. "
        "Dua belas sampel valid membentuk riwayat pengukuran. "
        "Dari riwayat tersebut dihitung 24 fitur statistik, lalu model pembelajaran mesin "
        "yang dipilih melalui Cross-Validation menghasilkan probabilitas dan kelas prediksi."
    )
    st.markdown("#### Penyimpanan Data")
    st.write(
        "Hasil pengukuran dan prediksi disimpan secara lokal di folder "
        "`riwayat_model_terbaik` pada komputer yang menjalankan aplikasi. "
        "Data tersebut dapat digunakan untuk meninjau kembali hasil monitoring."
    )
    st.markdown("#### Batasan Penggunaan")
    st.warning(
        "Prediksi menggambarkan **indikasi QoS jaringan** berdasarkan data pelatihan, "
        "bukan verifikasi langsung terhadap buffering atau gangguan video pada perangkat penerima. "
        "Throughput adalah lalu lintas upload adaptor, bukan hasil speed test internet."
    )
    st.caption("LiveStreamQoS | Aplikasi penelitian monitoring dan prediksi QoS jaringan Wi-Fi.")


def run_app():
    initialize_state()
    st.title("LiveStreamQoS")
    st.caption("Monitoring Wi-Fi dan prediksi indikasi kualitas jaringan 60 detik ke depan.")

    if not MODEL_FILE.exists():
        st.error(f"File model belum ditemukan: `{MODEL_FILE.name}`. Letakkan file .joblib hasil Colab di folder yang sama.")
        st.stop()
    try:
        bundle, selected_name, selected_model = cached_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
    except Exception as exc:
        st.error(f"Tidak dapat memuat model: {exc}")
        st.info("Pastikan Anda menggunakan file .joblib hasil Colab Cross-Validation dan library sesuai requirements.txt.")
        st.stop()

    if selected_name != "XGBoost":
        st.error("Berkas model tidak cocok dengan model terpilih dari Cross-Validation. "
                 "Gunakan file model .joblib yang disertakan.")
        st.stop()
    st.info("12 interval historis × 5 detik | 24 fitur Latency, Jitter, Packet Loss, dan Throughput "
            "| Prediksi indikasi QoS 60 detik berikutnya.")

    tabs = st.tabs(["Home", "Prediksi QoS", "About"])
    with tabs[0]:
        show_home()

    with tabs[1]:
        st.subheader("Monitoring Wi-Fi dan Hasil Prediksi")
        st.caption("1. Pilih adaptor Wi-Fi. 2. Klik Mulai Monitoring. "
                   "3. Amati hasil prediksi setelah 12 sampel valid.")
        nic_names = _nic_candidates()
        if not nic_names:
            st.error("Tidak ditemukan interface jaringan. Periksa koneksi dan adaptor Wi-Fi.")
            st.stop()
        sampler = st.session_state.best_sampler
        active = sampler is not None and sampler.thread.is_alive()
        nic = st.selectbox("Adaptor jaringan", options=nic_names, disabled=active,
                           help="Pilih adaptor Wi-Fi aktif. Throughput dihitung dari data yang dikirim adaptor ini.")
        start_c, stop_c = st.columns(2)
        if start_c.button("Mulai Monitoring", type="primary", use_container_width=True, disabled=active):
            try:
                _ping_command(PING_TARGET)
                sid = "CV-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]
                st.session_state.best_session_id = sid
                st.session_state.best_samples = []
                st.session_state.best_buffer = []
                st.session_state.best_prediction = None
                st.session_state.best_predictions = []
                st.session_state.best_last_note = ""
                st.session_state.best_sampling_error = ""
                sampler = NetworkSampler(nic, PING_TARGET, sid)
                st.session_state.best_sampler = sampler
                sampler.start()
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if stop_c.button("Hentikan Monitoring", use_container_width=True, disabled=not active):
            sampler.stop()
            st.session_state.best_sampler = None
            st.rerun()
        if active:
            st.caption(f"Status: aktif | Sesi: `{st.session_state.best_session_id}` | "
                       "Interval pengambilan: 5 detik")
        else:
            st.caption("Status: berhenti. Riwayat sesi terakhir tetap tampil dan CSV tersimpan.")
        live_monitor_panel(bundle, selected_name, selected_model)

    with tabs[2]:
        show_about()

if __name__ == "__main__":
    run_app()
