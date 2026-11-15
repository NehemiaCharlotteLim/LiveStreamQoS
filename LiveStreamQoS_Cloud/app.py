"""LiveStreamQoS, aplikasi cloud 3 halaman untuk evaluasi antarmuka (SUS).

Jika responden menjalankan sensor Windows di laptopnya, browser membaca
127.0.0.1 dengan izin eksplisit dan mengirim hasil QoS ke aplikasi untuk inferensi.
Tanpa sensor lokal, peserta dapat menggunakan data contoh atau unggah CSV.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import xgboost

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "model_bundle_qos_4parameter_CV_TERBAIK.joblib"
SENSOR_ZIP = BASE / "Sensor_WiFi_Windows.zip"
QOS = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]
STATS = ["last", "mean", "min", "max", "std", "delta"]
FEATURES = [f"{stat}_{name}" for stat in STATS for name in QOS]
HISTORY = 12

st.set_page_config(page_title="LiveStreamQoS | Prediksi QoS", page_icon="📶", layout="wide")

BRIDGE = st.components.v2.component(
    "livestreamqos_wifi_sensor_v2_autopair",
    html="""
      <section class="lsq-bridge">
        <div class="lsq-actions">
          <button class="lsq-connect" type="button">Mulai Monitoring</button>
          <button class="lsq-disconnect" type="button">Berhenti</button>
        </div>
        <p class="lsq-status">Sensor belum terhubung. Jalankan peluncur Windows untuk membuka website dan memulai pengukuran secara otomatis.</p>
      </section>
    """,
    css="""
      .lsq-bridge {font-family: inherit; color: inherit; padding: 6px 0 2px;}
      .lsq-actions {display:flex; gap:12px; flex-wrap:wrap;}
      .lsq-actions button {border:1px solid #8b97a9; color:inherit; background:transparent;
         padding:10px 16px; cursor:pointer; border-radius:8px; font-weight:600;}
      .lsq-actions .lsq-connect {background:#D95764; border-color:#D95764; color:white;}
      .lsq-actions button:disabled {opacity:.45; cursor:default;}
      .lsq-status {font-size:.89rem; opacity:.9; margin:10px 0 0; line-height:1.4;}
    """,
    js=(BASE / "bridge.js").read_text(encoding="utf-8"),
)


# Pengukur HTTP langsung di browser HP. Tidak menggunakan fitur/prediksi model XGBoost.
HP_PROBE = st.components.v2.component(
    "livestreamqos_mobile_http_probe_v1",
    html="""
      <section class="lsq-mobile">
        <div class="lsq-m-actions">
          <button class="lsq-m-start" type="button">Mulai Monitoring HP</button>
          <button class="lsq-m-stop" type="button" disabled>Berhenti</button>
          <button class="lsq-m-csv" type="button" disabled>Simpan CSV</button>
        </div>
        <div class="lsq-m-progress"><div class="lsq-m-bar-fill"></div></div>
        <p class="lsq-m-counter"><span class="lsq-m-count">0/12 pengukuran</span> · <span class="lsq-m-activity">Belum dimulai</span></p>
        <div class="lsq-m-metrics">
          <div><small>Respons HTTP</small><strong class="lsq-m-latency">—</strong></div>
          <div><small>Variasi respons</small><strong class="lsq-m-jitter">—</strong></div>
          <div><small>Permintaan gagal</small><strong class="lsq-m-failure">—</strong></div>
          <div><small>Estimasi unduh</small><strong class="lsq-m-download">—</strong></div>
        </div>
        <div class="lsq-m-result"><small>Status koneksi (indikatif)</small><strong class="lsq-m-label">Belum tersedia</strong><p class="lsq-m-reason">Klik Mulai Monitoring HP.</p></div>
        <p class="lsq-m-status" role="status">Pengujian browser memakai permintaan HTTP dari koneksi HP ke server website.</p>
        <p class="lsq-m-probe-info"></p>
      </section>
    """,
    css="""
       .lsq-mobile {font-family:inherit; color:inherit; padding:8px 1px 4px;}
       .lsq-m-actions {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;}
       .lsq-m-actions button {padding:10px 12px; border-radius:9px; border:1px solid #718096; background:transparent; color:inherit; font-weight:600; cursor:pointer; flex-grow:1;}
       .lsq-m-actions .lsq-m-start {background:#D95764; border-color:#D95764; color:white;}
       .lsq-m-actions button:disabled {opacity:.5; cursor:default;}
       .lsq-m-progress {height:8px; background:rgba(127,127,127,.22); border-radius:5px; overflow:hidden;}
       .lsq-m-bar-fill {background:#D95764; height:100%; width:0%; transition:width .25s;}
       .lsq-m-counter {font-size:.85rem; margin:6px 0 12px; opacity:.8;}
       .lsq-m-metrics {display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px;}
       .lsq-m-metrics div {border-radius:11px; padding:12px; border:1px solid rgba(130,145,160,.35); background:rgba(125,125,125,.07); min-width:0;}
       .lsq-m-metrics small {display:block; opacity:.75; font-size:.78rem; margin-bottom:5px;}
       .lsq-m-metrics strong {font-size:1.14rem; word-break:break-word;}
       .lsq-m-result {border:1px solid rgba(130,145,160,.35); border-radius:11px; padding:15px; margin:13px 0;}
       .lsq-m-result small {display:block; font-size:.82rem; opacity:.8; margin-bottom:4px;}
       .lsq-m-label {display:block; font-size:1.8rem; font-weight:750;}
       .lsq-m-good {color:#28B487;} .lsq-m-bad {color:#F16B7C;}
       .lsq-m-reason {font-size:.88rem; margin:6px 0 0;}
       .lsq-m-status {font-size:.88rem; line-height:1.5;}
       .lsq-m-probe-info {opacity:.7; font-size:.75rem;}
       @media(min-width:630px) {.lsq-m-metrics {grid-template-columns:repeat(4,minmax(0,1fr));}}
    """,
    js=(BASE / "browser_probe.js").read_text(encoding="utf-8"),
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_FILE.is_file():
        raise RuntimeError("File model tidak ditemukan di folder website.")
    bundle = joblib.load(MODEL_FILE)  # model dari developer tepercaya, jangan unpickle upload pengguna
    if not isinstance(bundle, dict) or bundle.get("feature_version") != "qos_24_summary_v1":
        raise RuntimeError("Versi fitur model tidak sesuai.")
    if bundle.get("features") != FEATURES or int(bundle.get("history_intervals", 0)) != HISTORY:
        raise RuntimeError("Jumlah/urutan fitur model tidak sesuai dengan aplikasi.")
    if bundle.get("best_model_name") != "XGBoost":
        raise RuntimeError("Bundle bukan model terbaik yang diharapkan.")
    for key, actual in [("sklearn_version", sklearn.__version__), ("xgboost_version", xgboost.__version__)]:
        if bundle.get(key) and str(bundle[key]) != actual:
            raise RuntimeError(f"Library tidak sesuai ({key}: {actual}). Periksa requirements.txt")
    model = bundle.get("model")
    if not hasattr(model, "predict_proba") or 1 not in list(model.classes_):
        raise RuntimeError("Model belum bisa menghasilkan probabilitas Gangguan.")
    return bundle, model


def clean_samples(raw: pd.DataFrame) -> pd.DataFrame:
    """Terima 12 interval terakhir, tidak memalsukan sampel hilang."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Data pengukuran kosong.")
    if not set(QOS).issubset(raw.columns):
        missing = [x for x in QOS if x not in raw.columns]
        raise ValueError("CSV belum memiliki kolom: " + ", ".join(missing))
    if len(raw) < HISTORY:
        raise ValueError(f"Diperlukan {HISTORY} sampel. Saat ini baru tersedia {len(raw)}.")
    data = raw.tail(HISTORY).copy().reset_index(drop=True)
    if "status_pengukuran" in data.columns:
        if not data["status_pengukuran"].astype(str).str.lower().eq("ok").all():
            raise ValueError("Ada pembacaan ping tidak lengkap pada 12 sampel terakhir. Tunggu pengukuran berikutnya.")
    if "session_id" in data.columns and data["session_id"].nunique(dropna=False) != 1:
        raise ValueError("12 sampel berasal dari sesi yang berbeda. Tidak boleh digabungkan dalam satu window.")
    for col in QOS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    mat = data[QOS].to_numpy(dtype=float)
    if not np.isfinite(mat).all() or (mat < 0).any():
        raise ValueError("12 pengukuran harus berupa angka QoS yang lengkap dan tidak negatif.")
    if (data["packet_loss_percent"] > 100).any():
        raise ValueError("Packet Loss harus berada pada rentang 0 sampai 100%.")
    if "interval_start" in data.columns:
        ts = pd.to_datetime(data["interval_start"], errors="coerce")
        if ts.isna().any():
            raise ValueError("Timestamp interval_start tidak dapat dibaca.")
        gaps = ts.diff().dt.total_seconds().iloc[1:]
        if not gaps.between(3.5, 8.0).all():
            raise ValueError("Interval tidak berurutan sekitar 5 detik. Model memerlukan 12 interval kontinu.")
    return data


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    matrix = data[QOS].to_numpy(dtype=float)
    features = {}
    for stat in STATS:
        for i,col in enumerate(QOS):
            v = matrix[:,i]
            if stat == "last": value=v[-1]
            elif stat == "mean": value=v.mean()
            elif stat == "min": value=v.min()
            elif stat == "max": value=v.max()
            elif stat == "std": value=v.std(ddof=0)
            else: value=v[-1]-v[0]
            features[f"{stat}_{col}"] = float(value)
    return pd.DataFrame([features], columns=FEATURES)


def predict(data: pd.DataFrame, bundle, model):
    probabilities = model.predict_proba(make_features(data))[0]
    class_ids = list(model.classes_)
    prob_bad = float(probabilities[class_ids.index(1)])
    threshold = float(bundle.get("threshold",0.5))
    return ("Gangguan" if prob_bad >= threshold else "Normal", prob_bad)


def show_prediction(raw: pd.DataFrame, bundle, model, *, show_table=True):
    if raw.empty:
        st.info("Pengukuran belum tersedia.")
        return
    last = raw.iloc[-1]
    metrics=st.columns(4)
    for col, title, unit, widget in zip(QOS, ["Latency", "Jitter", "Packet Loss", "Throughput"], ["ms", "ms", "%", "Mbps"], metrics):
        try:
            v=float(last[col])
            s=f"{v:.3f} {unit}" if math.isfinite(v) else "Belum tersedia"
        except (KeyError, ValueError, TypeError):
            s="Belum tersedia"
        widget.metric(title,s)
    trailing_good = 0
    for _, rec in raw.tail(HISTORY).iloc[::-1].iterrows():
        values = pd.to_numeric(rec[QOS], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or str(rec.get("status_pengukuran", "ok")).lower() != "ok":
            break
        trailing_good += 1
    st.progress(min(1.0,trailing_good/HISTORY),text=f"Riwayat valid berurutan: {trailing_good}/{HISTORY} interval")
    try:
        cleaned = clean_samples(raw)
    except ValueError as exc:
        st.info(f"Prediksi belum tersedia: {exc}")
        return
    label, p_bad = predict(cleaned,bundle,model)
    a,b,c=st.columns(3)
    a.metric("Prediksi kondisi QoS",label)
    b.metric("Probabilitas Gangguan",f"{p_bad*100:.2f}%")
    c.metric("Probabilitas Normal",f"{(1-p_bad)*100:.2f}%")
    st.caption("Prediksi indikatif berdasarkan 12 × 5 detik riwayat QoS untuk horizon berikutnya sekitar 60 detik. Bukan kepastian terjadinya buffering video.")
    if show_table:
        with st.expander("Lihat 12 pengukuran yang digunakan"):
            st.dataframe(cleaned[QOS].rename(columns={
                "latency_ms":"Latency (ms)","jitter_ms":"Jitter (ms)","packet_loss_percent":"Packet Loss (%)","throughput_mbps":"Throughput (Mbps)"
            }), hide_index=True, use_container_width=True)


def home():
    st.subheader("Selamat Datang di LiveStreamQoS")
    st.write("LiveStreamQoS membantu pengguna memeriksa kondisi koneksi internet dan memahami status indikatif **Normal/Gangguan**. Untuk laptop Windows, prediksi memakai model penelitian yang dilatih dari empat parameter QoS.")
    st.markdown("#### Menguji dari HP tanpa instalasi")
    st.markdown("1. Buka link website LiveStreamQoS melalui Chrome atau Safari di HP.\n2. Masuk ke **Prediksi QoS**, lalu pilih **HP / Browser**.\n3. Tekan **Mulai Monitoring HP**.\n4. Status **Normal/Gangguan** muncul dari pemeriksaan HTTP. Rangkuman menggunakan hingga 12 pembacaan.\n5. Isi kuesioner SUS melalui link terpisah dari peneliti.")
    st.info("Pengukuran HP dilakukan melalui HTTP ke server website. Ini adalah **indikasi koneksi browser**, bukan prediksi model XGBoost dan bukan pengukuran ICMP/adapter Wi-Fi.")
    with st.expander("Menggunakan aplikasi dari laptop Windows"):
        st.markdown("Unduh sensor Windows di halaman Prediksi QoS, ekstrak, lalu jalankan **MULAI_MONITORING.bat**. Browser akan terbuka otomatis dan menjalankan prediksi model menggunakan 12 interval QoS, tanpa memasukkan kode manual.")



def prediction_tab(bundle,model):
    st.subheader("Monitoring Koneksi dan Hasil")
    mode=st.radio("Pilih cara pengujian", [
        "HP / Browser (langsung, tanpa instalasi)",
        "Laptop Windows (sensor lokal)",
        "Unggah CSV atau contoh data",
    ],horizontal=False)
    if mode.startswith("HP"):
        st.write("**Monitoring koneksi dari browser HP Anda.** Tidak memerlukan sensor atau aplikasi tambahan.")
        st.caption("Hasil **Normal/Gangguan** di sini berdasarkan pemeriksaan HTTP dan aturan indikatif browser, bukan keluaran model XGBoost. Pengujian ini menggunakan koneksi internet aktif HP, baik Wi-Fi maupun seluler.")
        HP_PROBE(data={}, key="lsq_mobile_probe")
        st.caption("Pengukuran melalui server website mencakup jalur internet, sehingga gangguan server atau jaringan di luar Wi-Fi HP juga dapat memengaruhi hasil. Batas status eksploratif: permintaan gagal ≥25%, respons rata-rata >350 ms, variasi >100 ms, atau estimasi unduh <1,5 Mbps. Bukan standar resmi atau hasil validasi model.")
        return
    if mode.startswith("Laptop"):

        st.write("**Monitoring Wi-Fi laptop Windows Anda.** Sensor berjalan di laptop, bukan di server website.")
        if SENSOR_ZIP.exists():
            st.download_button("Unduh Sensor Wi-Fi Windows (setup sekali saja)", SENSOR_ZIP.read_bytes(), file_name="Sensor_WiFi_Windows.zip", mime="application/zip", use_container_width=True)
        with st.expander("Cara memulai, tanpa menyalin kode", expanded=False):
            st.markdown("1. Ekstrak ZIP sensor, lalu klik dua kali **MULAI_MONITORING.bat**.\n2. Sensor otomatis membuka halaman **Prediksi QoS** di browser laptop yang sama.\n3. Browser mencoba menghubungkan sensor. Jika meminta izin mengakses jaringan lokal/loopback, pilih **Izinkan**. Jika tidak otomatis, klik **Mulai Monitoring**.\n4. Lihat hasil setelah 12 sampel QoS valid terkumpul. **Berhenti** hanya memutus tampilan; tutup jendela sensor untuk menghentikan pengukuran.")
        st.caption("Untuk keamanan, sensor hanya dapat diakses dari browser website resmi dan memakai kode sesi otomatis di perangkat Anda. Kode tidak perlu diketik atau dikirim melalui kolom website.")
        result=BRIDGE(data={},default={"enabled":False,"snapshot":None},key="wifi_bridge_oneclick",on_enabled_change=lambda:None,on_snapshot_change=lambda:None)
        reading = getattr(result,"snapshot",None)
        if not isinstance(reading,dict) or reading.get("schema") != "lsq_sensor_v1":
            st.info("Belum menerima pengukuran dari sensor. Jika sensor sudah berjalan, klik **Mulai Monitoring** di atas dan izinkan akses jaringan lokal bila diminta browser.")
            return
        rows=reading.get("rows",[])
        if not isinstance(rows,list) or len(rows)>HISTORY:
            st.warning("Data sensor tidak sesuai. Pastikan sensor menggunakan versi terbaru.")
            return
        st.caption(f"Sumber: Wi-Fi laptop Anda | Sesi: {reading.get('session_id','-')} | Interval: {reading.get('sequence',0)}")
        if reading.get("error"):
            st.caption("Status pembacaan terakhir: " + str(reading["error"]))
        show_prediction(pd.DataFrame(rows),bundle,model)
    else:
        st.caption("Mode cadangan. Hasil didasarkan pada CSV historis atau contoh pengukuran, bukan pemantauan Wi-Fi browser secara langsung.")
        choice=st.radio("Jenis data",["Unggah CSV pengukuran", "Gunakan contoh historis"],horizontal=True)
        if choice.startswith("Unggah"):
            uploader=st.file_uploader("Unggah file CSV dari sensor atau pengukuran sendiri",type=["csv"],help="Minimal 12 baris terakhir berurutan sekitar 5 detik dengan empat parameter QoS.")
            if uploader is None:
                st.info("Gunakan `riwayat_qos_sensor.csv` dari folder sensor setelah 12 interval terkumpul.")
                return
            try:
                raw=pd.read_csv(uploader,nrows=20000)
            except Exception as exc:
                st.error(f"CSV belum dapat dibaca: {exc}")
                return
        else:
            variant=st.selectbox("Pilih contoh",["Contoh 1","Contoh 2"])
            raw=pd.read_csv(BASE/("contoh_qos_1.csv" if variant == "Contoh 1" else "contoh_qos_2.csv"))
            st.info("Contoh digunakan untuk mencoba antarmuka dan bukan hasil pengukuran Wi-Fi pengunjung.")
        show_prediction(raw,bundle,model)


def about():
    st.subheader("Tentang LiveStreamQoS")
    st.write("LiveStreamQoS merupakan aplikasi penelitian untuk memeriksa kondisi jaringan dan menampilkan indikasi Normal/Gangguan.")
    st.markdown("#### Dua jenis pengukuran")
    st.markdown("**Laptop Windows:** sensor lokal mengukur empat parameter QoS (Latency ICMP, variasi RTT, Packet Loss ICMP, dan outbound Throughput) setiap sekitar lima detik. Model terlatih merangkum 12 interval menjadi 24 fitur dan memperkirakan indikasi QoS 60 detik ke depan.\n\n**HP / Browser:** website mengukur waktu respons HTTP, variasi respons, kegagalan permintaan HTTP, dan perkiraan kecepatan unduh. Status Normal/Gangguan berasal dari aturan sederhana berdasarkan pengukuran browser saat itu, **bukan prediksi model terlatih**.")
    st.markdown("#### Manfaat aplikasi\n- Membantu pengguna memahami kestabilan koneksi saat mengakses layanan internet.\n- Menampilkan status indikatif yang mudah dibaca.\n- Menyediakan alternatif pengujian melalui HP tanpa memasang sensor Windows.")
    st.markdown("#### Privasi dan batasan\n- Pengukuran HP dijalankan di browser dengan permintaan HTTP ke server website. Tidak meminta kata sandi Wi-Fi.\n- Data pengukuran HP disimpan sementara di browser; pengguna bisa mengunduh CSV sendiri. Permintaan HTTP ke server tetap dapat tercatat dalam log penyedia hosting.\n- Status browser bukan bukti buffering video dan tidak dapat disamakan dengan akurasi model penelitian.\n- Pengguna laptop Windows dapat memakai sensor lokal untuk metode penelitian yang menggunakan empat parameter QoS.")
    st.caption("Kuesioner System Usability Scale (SUS) disediakan terpisah oleh peneliti.")



def main():
    st.title("LiveStreamQoS")
    st.caption("Monitoring dan prediksi kondisi Quality of Service jaringan")
    try:
        bundle, model=load_model()
    except Exception as exc:
        st.error(f"Model belum dapat dimuat: {exc}")
        st.stop()
    home_tab, pred_tab, about_tab = st.tabs(["Home", "Prediksi QoS", "About"], default="Prediksi QoS" if st.query_params.get("monitor") == "1" else "Home")
    with home_tab: home()
    with pred_tab: prediction_tab(bundle, model)
    with about_tab: about()


if __name__ == "__main__":
    main()
