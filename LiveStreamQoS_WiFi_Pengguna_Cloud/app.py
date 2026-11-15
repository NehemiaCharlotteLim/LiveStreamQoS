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
    "livestreamqos_wifi_sensor_v1",
    html="""
      <section class="lsq-bridge">
        <div class="lsq-actions">
          <button class="lsq-connect" type="button">Hubungkan Sensor Wi-Fi</button>
          <button class="lsq-disconnect" type="button">Putuskan</button>
        </div>
        <p class="lsq-status">Belum terhubung. Jalankan sensor Windows, salin kode, lalu tekan Hubungkan.</p>
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
    st.write("LiveStreamQoS membantu pengguna memahami kualitas jaringan melalui empat parameter QoS dan prediksi indikasi **Normal/Gangguan** untuk periode sekitar 60 detik berikutnya.")
    a,b=st.columns(2)
    with a:
        st.markdown("**Latency (ms)**: rata-rata waktu pulang-pergi paket uji.")
        st.markdown("**Jitter (ms)**: variasi waktu respons antarbalasan ping.")
    with b:
        st.markdown("**Packet Loss (%)**: paket ping yang tidak memperoleh respons.")
        st.markdown("**Throughput (Mbps)**: jumlah data yang keluar dari adaptor Wi-Fi per detik.")
    st.markdown("#### Cara mencoba Wi-Fi laptop sendiri")
    st.markdown("1. Pada tab **Prediksi QoS**, pilih **Wi-Fi laptop saya**.\n2. Unduh dan jalankan **Sensor Wi-Fi Windows** di laptop yang sama.\n3. Salin kode penghubung dari CMD sensor ke website, lalu klik **Hubungkan Sensor Wi-Fi**.\n4. Jika browser meminta izin akses jaringan lokal, klik **Izinkan**.\n5. Tunggu 12 interval valid (sekitar satu menit) dan amati prediksi.\n6. Setelah mencoba aplikasi, isi kuesioner SUS melalui tautan terpisah dari peneliti.")
    st.info("Untuk HP atau jika sensor Windows tidak dapat berjalan, gunakan opsi data contoh/unggah CSV. Browser saja tidak bisa mengambil ICMP ping dan counter unggah adaptor pengguna.")


def prediction_tab(bundle,model):
    st.subheader("Monitoring Wi-Fi dan Hasil Prediksi")
    mode=st.radio("Sumber data pengukuran",["Wi-Fi laptop saya (sensor lokal)","Unggah CSV atau contoh data"],horizontal=True)
    if mode.startswith("Wi-Fi"):
        st.write("Pengukuran dilakukan oleh sensor di **laptop Windows milik Anda sendiri**, bukan server website. Tanpa sensor, website tidak dapat membaca adaptor Wi-Fi laptop.")
        if SENSOR_ZIP.exists():
            st.download_button("Unduh Sensor Wi-Fi Windows (.zip)",SENSOR_ZIP.read_bytes(),file_name="Sensor_WiFi_Windows.zip",mime="application/zip",use_container_width=True)
        with st.expander("Petunjuk membuka sensor",expanded=True):
            st.markdown("1. Ekstrak ZIP sensor.\n2. Klik dua kali **JALANKAN_SENSOR_WINDOWS.bat** pada laptop Windows yang terhubung Wi-Fi. Pastikan Python terpasang.\n3. Tunggu kode penghubung muncul di CMD. Jangan tutup CMD.\n4. Masukkan kode tersebut di bawah ini.\n5. Klik Hubungkan. Jika browser bertanya izin **akses perangkat lokal / loopback**, izinkan.\n6. Pengukuran akan muncul setiap 5 detik, prediksi ketika 12 sampel valid berurutan tersedia.")
        code=st.text_input("Kode penghubung dari CMD sensor",type="password",placeholder="Tempel kode dari jendela sensor",help="Kode berlaku selama satu sesi sensor. Tidak perlu memasukkan nama atau password Wi-Fi.")
        if not code:
            st.info("Jalankan sensor dan masukkan kodenya untuk menghubungkan pengukuran Wi-Fi Anda.")
            return
        result=BRIDGE(data={"token": code.strip()},default={"enabled":False,"snapshot":None},key="wifi_bridge",on_enabled_change=lambda:None,on_snapshot_change=lambda:None)
        reading = getattr(result,"snapshot",None)
        if not isinstance(reading,dict) or reading.get("schema") != "lsq_sensor_v1":
            st.info("Belum menerima data. Klik **Hubungkan Sensor Wi-Fi** dan izinkan browser mengakses 127.0.0.1.")
            return
        rows=reading.get("rows",[])
        if not isinstance(rows,list) or len(rows)>HISTORY:
            st.warning("Data sensor tidak sesuai. Pastikan menggunakan sensor versi terbaru.")
            return
        # Data diterima melalui browser pengguna, bukan ping yang dijalankan cloud.
        st.caption(f"Sumber: sensor di laptop pengguna | Session: {reading.get('session_id','-')} | Pembacaan: {reading.get('sequence',0)}")
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
    st.write("LiveStreamQoS adalah aplikasi penelitian untuk membantu memantau empat parameter Quality of Service (QoS) jaringan dan memperkirakan perubahan kondisi QoS dalam dua kategori: Normal dan Gangguan.")
    st.markdown("#### Kegunaan\n- Menampilkan pengukuran Latency, Jitter, Packet Loss, dan Throughput.\n- Menampilkan probabilitas indikatif Normal/Gangguan setelah 12 interval.\n- Membantu pengguna memahami kondisi koneksi saat menggunakan layanan internet.")
    st.markdown("#### Cara kerja\nSensor lokal mengukur QoS setiap sekitar 5 detik. Sebanyak 12 pembacaan lengkap dan berurutan diringkas menjadi 24 fitur statistik. Model yang terlatih memproses fitur tersebut untuk menghasilkan probabilitas kondisi QoS pada horizon berikutnya.")
    st.markdown("#### Privasi dan batasan\n- Sensor hanya mendengarkan di **127.0.0.1**, menggunakan kode sesi sementara.\n- Pengukuran lokal disimpan pada CSV di laptop pengguna.\n- Setelah pengguna menghubungkan sensor, 12 pengukuran dikirim ke server Streamlit untuk dihitung prediksinya.\n- Kode sensor **tidak** meminta password Wi-Fi atau merekam aktivitas browser.\n- Pengukuran berbasis ICMP berbeda dari paket media streaming. Hasil indikatif tidak otomatis membuktikan gangguan tayangan.\n- HP tidak didukung untuk sensor Windows; gunakan data contoh atau CSV bila hanya memakai HP.")
    st.caption("Pengujian SUS dilakukan melalui kuesioner terpisah yang dibagikan peneliti.")


def main():
    st.title("LiveStreamQoS")
    st.caption("Monitoring dan prediksi kondisi Quality of Service jaringan")
    try:
        bundle, model=load_model()
    except Exception as exc:
        st.error(f"Model belum dapat dimuat: {exc}")
        st.stop()
    home_tab, pred_tab, about_tab = st.tabs(["Home", "Prediksi QoS", "About"])
    with home_tab: home()
    with pred_tab: prediction_tab(bundle, model)
    with about_tab: about()


if __name__ == "__main__":
    main()
