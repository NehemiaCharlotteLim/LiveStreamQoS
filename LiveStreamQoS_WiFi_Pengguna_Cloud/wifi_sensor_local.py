"""Sensor lokal LiveStreamQoS. Windows only. Tidak memerlukan OBS.

Pengukuran terjadi di PC pengguna. Server HTTP hanya di 127.0.0.1:8765,
dilindungi token sesi dan izin CORS terbatas; tidak mengekspor kunci Wi-Fi.
"""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

try:
    import psutil
except ImportError:
    raise SystemExit("psutil belum tersedia. Jalankan: python -m pip install psutil")

if sys.platform != "win32":
    raise SystemExit("Sensor ini khusus laptop Windows. Pengunjung HP perlu opsi unggah CSV.")

PORT = 8765
PING_TARGET = "1.1.1.1"
INTERVAL = 5.0
PING_COUNT = 4
PING_TIMEOUT_MS = 800
SECRET = secrets.token_urlsafe(18)
DATA = deque(maxlen=12)
DATA_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
LOGFILE = Path(__file__).resolve().parent / "riwayat_qos_sensor.csv"
SESSION = datetime.now().strftime("WS%Y%m%d_%H%M%S")
SEQ = 0
INTERFACE = None
LAST_ERROR = ""
COLS = ["interval_start", "interval_end", "session_id", "interface_name", "latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps", "interval_seconds", "status_pengukuran", "ping_error"]


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def pick_wifi():
    stats = psutil.net_if_stats()
    io = psutil.net_io_counters(pernic=True)
    found = [name for name, state in stats.items()
             if state.isup and name in io and
             any(w in name.lower() for w in ("wi-fi", "wifi", "wireless", "wlan")) and
             not any(w in name.lower() for w in ("direct", "virtual", "bluetooth"))]
    found.sort(key=lambda name: (name.lower() != "wi-fi", name.lower()))
    return found[0] if found else None


def sent_bytes(interface):
    c = psutil.net_io_counters(pernic=True).get(interface)
    return c.bytes_sent if c else None


def parse_ping(text):
    # Windows versi Inggris dan sebagian lokalisasi Indonesia.
    times = re.findall(r"(?:time|waktu)\s*[=<]\s*(\d+(?:[.,]\d+)?)\s*(?:ms|msec)\b", text, flags=re.I)
    rtts = [float(x.replace(",", ".")) for x in times]
    summary = re.search(r"(?:Sent|Dikirim|Terkirim)\s*=\s*(\d+)\s*,\s*(?:Received|Diterima)\s*=\s*(\d+)\s*,\s*(?:Lost|Hilang)\s*=\s*(\d+)", text, flags=re.I | re.S)
    if summary:
        sent = int(summary.group(1))
        received = int(summary.group(2))
    else:
        sent = PING_COUNT
        received = len(rtts) if rtts else None
    latency = sum(rtts)/len(rtts) if rtts else None
    jitter = sum(abs(b-a) for a, b in zip(rtts, rtts[1:]))/(len(rtts)-1) if len(rtts) > 1 else None
    loss = 100.0 * (sent - received)/sent if received is not None and sent > 0 else None
    return latency, jitter, loss


def sample_once(interface):
    started = now_iso()
    t0 = time.monotonic()
    b0 = sent_bytes(interface)
    err = ""
    output = ""
    try:
        proc = subprocess.Popen(["ping", "-n", str(PING_COUNT), "-w", str(PING_TIMEOUT_MS), PING_TARGET], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        time.sleep(max(0.0, INTERVAL - (time.monotonic() - t0)))
        try:
            out, stderr = proc.communicate(timeout=0.15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, stderr = proc.communicate()
            err = "Ping belum selesai dalam 5 detik."
        output = (out or "") + "\n" + (stderr or "")
    except OSError as exc:
        err = f"Ping tidak dapat dijalankan: {exc}"
        time.sleep(max(0.0, INTERVAL - (time.monotonic() - t0)))
    ended = now_iso()
    t1 = time.monotonic()
    b1 = sent_bytes(interface)
    duration = t1 - t0
    latency, jitter, loss = parse_ping(output)
    throughput = ((b1 - b0)*8.0 / duration / 1e6) if b0 is not None and b1 is not None and b1 >= b0 and duration > 0 else None
    vals = [latency, jitter, loss, throughput]
    valid = not err and 3.5 <= duration <= 7.0 and all(x is not None and math.isfinite(x) and x >= 0 for x in vals)
    if not valid and not err:
        err = "RTT/Jitter/Packet Loss atau counter Wi-Fi belum lengkap."
    return {
        "interval_start":started, "interval_end":ended,"session_id":SESSION,"interface_name":interface,
        "latency_ms":round(latency,3) if latency is not None else None,
        "jitter_ms":round(jitter,3) if jitter is not None else None,
        "packet_loss_percent":round(loss,3) if loss is not None else None,
        "throughput_mbps":round(throughput,6) if throughput is not None else None,
        "interval_seconds":round(duration,3),"status_pengukuran":"ok" if valid else "partial",
        "ping_error":err,
    }


def sensor_loop():
    global SEQ, LAST_ERROR
    existed = LOGFILE.exists() and LOGFILE.stat().st_size > 0
    with LOGFILE.open("a", encoding="utf-8" if existed else "utf-8-sig", newline="", buffering=1) as fh:
        if existed:
            with LOGFILE.open("r", encoding="utf-8-sig", newline="") as r:
                existing_header = next(csv.reader(r), [])
            if existing_header != COLS:
                raise RuntimeError("Format CSV lama berbeda. Pindahkan/hapus riwayat_qos_sensor.csv lalu mulai ulang.")
        writer = csv.DictWriter(fh, fieldnames=COLS)
        if not existed:
            writer.writeheader()
        while not STOP_EVENT.is_set():
            try:
                record = sample_once(INTERFACE)
                writer.writerow(record)
                fh.flush()
                with DATA_LOCK:
                    DATA.append(record)
                    SEQ += 1
                    LAST_ERROR = record.get("ping_error", "")
                label = "OK" if record["status_pengukuran"] == "ok" else "PARTIAL"
                print(f'[{SEQ:03d}] {record["interval_end"]} {label} | Latency={record["latency_ms"]} Jitter={record["jitter_ms"]} Loss={record["packet_loss_percent"]} Throughput={record["throughput_mbps"]}', flush=True)
            except Exception as exc:
                LAST_ERROR = str(exc)
                print("Gagal membaca interval:", exc, file=sys.stderr, flush=True)
                if STOP_EVENT.wait(INTERVAL):
                    return


def authorized_origin(origin):
    if not origin:
        return False
    u = urlsplit(origin)
    # Izinkan hanya aplikasi Streamlit HTTPS dan versi lokal pengembang.
    return ((u.scheme == "https" and bool(re.fullmatch(r"[a-z0-9-]+\.streamlit\.app", (u.hostname or "")))) or
            (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1") and u.port == 8501))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # hindari menampilkan token ke log

    def respond_headers(self, code=200, content_type="application/json", size=0):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin", "")
        if authorized_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "X-LSQ-Token, Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        if not authorized_origin(self.headers.get("Origin", "")):
            self.respond_headers(403)
        else:
            self.respond_headers(204)

    def do_GET(self):
        origin = self.headers.get("Origin", "")
        # Halaman cloud HARUS memiliki origin yang valid. Komunikasi tanpa
        # origin (mis. curl) tidak mendapat data untuk meminimalkan penyalahgunaan.
        if not authorized_origin(origin):
            self._json({"error": "Asal situs tidak diizinkan."}, 403)
            return
        if self.headers.get("X-LSQ-Token", "") != SECRET:
            self._json({"error": "Kode sensor salah. Salin kode di jendela CMD."}, 401)
            return
        if self.path.split("?",1)[0] != "/v1/snapshot":
            self._json({"error": "Endpoint tidak ditemukan."}, 404)
            return
        with DATA_LOCK:
            payload = {"schema": "lsq_sensor_v1", "session_id":SESSION, "sequence":SEQ,
                       "interval_seconds": INTERVAL, "rows": list(DATA),
                       "total_available": len(DATA), "error":LAST_ERROR}
        self._json(payload)

    def _json(self, data, code=200):
        buf=json.dumps(data, allow_nan=False).encode("utf-8")
        self.respond_headers(code, size=len(buf))
        self.wfile.write(buf)


def main():
    global INTERFACE
    INTERFACE = pick_wifi()
    if not INTERFACE:
        print("Wi-Fi aktif tidak terdeteksi. Hubungkan laptop ke Wi-Fi, lalu jalankan lagi.")
        print("Adapter tersedia:", ", ".join(psutil.net_if_stats().keys()))
        raise SystemExit(1)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        raise SystemExit(f"Tidak dapat memakai port {PORT}. Tutup jendela sensor lain. Detail: {exc}")
    print("\n=== SENSOR WIFI LOKAL | LIVESTREAMQOS ===")
    print("Mengukur Wi-Fi laptop ini, bukan jaringan server Streamlit.")
    print(f"Adapter          : {INTERFACE}")
    print(f"Kode penghubung  : {SECRET}")
    print(f"Alamat lokal     : http://127.0.0.1:{PORT}")
    print(f"Data tersimpan   : {LOGFILE}")
    print("Interval         : 5 detik | 12 sampel berurutan untuk prediksi")
    print("Buka website yang dibagikan peneliti pada browser di LAPTOP YANG SAMA.")
    print("Isi kode penghubung di tab Prediksi QoS dan klik Hubungkan Sensor.")
    print("Jika browser meminta izin akses perangkat lokal, pilih Izinkan.")
    print("Sensor tidak membuka port ke internet. Hentikan dengan Ctrl+C.\n",flush=True)
    worker=threading.Thread(target=sensor_loop,daemon=True)
    worker.start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nSensor dihentikan. Data CSV sebelumnya tetap disimpan.")
    finally:
        STOP_EVENT.set()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
