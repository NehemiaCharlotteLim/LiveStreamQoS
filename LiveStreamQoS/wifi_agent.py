"""Agen pengukur QoS WiFi laptop untuk LiveStreamQoS online.

Berjalan HANYA pada 127.0.0.1, bukan server publik.
Menyediakan data pengukuran nyata melalui browser pengguna.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

try:
    import psutil
except ImportError:
    sys.exit("Library psutil belum tersedia. Jalankan: python -m pip install psutil")

HOST = "127.0.0.1"
APP_ORIGINS = {"https://livestreamqos.streamlit.app", "http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8502", "http://127.0.0.1:8502"}
PORT = 8765
TARGET = "1.1.1.1"
PERIOD = 5.0
PING_N = 4
VALID_KEYS = ("latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps")


def allowed_origin(origin: str | None) -> bool:
    """Batasi akses hanya pada deployment LiveStreamQoS yang diizinkan."""
    return origin in APP_ORIGINS


def wifi_adapters() -> list[str]:
    stats = psutil.net_if_stats()
    names = []
    for name, info in stats.items():
        low = name.lower()
        if info.isup and any(token in low for token in ("wi-fi", "wifi", "wlan", "wireless", "wlp")):
            names.append(name)
    return names


def ping_probe() -> dict:
    """ICMP bila ping tersedia; cadangan TCP mengukur kegagalan koneksi, bukan packet loss ICMP."""
    binary = shutil.which("ping")
    if binary:
        if sys.platform.startswith("win"):
            cmd = [binary, "-n", str(PING_N), "-w", "800", TARGET]
        elif sys.platform == "darwin":
            cmd = [binary, "-c", str(PING_N), "-W", "800", TARGET]
        else:
            cmd = [binary, "-c", str(PING_N), "-W", "1", TARGET]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=4.7)
            out = (completed.stdout or "") + "\n" + (completed.stderr or "")
            times = [float(v.replace(",", ".")) for v in re.findall(
                r"(?:time|waktu)\s*[=<]\s*(\d+(?:[.,]\d+)?)\s*(?:msec|ms|md)\b", out, re.IGNORECASE)]
            win = re.search(r"(?:Sent|Dikirim|Terkirim)\s*=\s*(\d+)\s*,\s*(?:Received|Diterima)\s*=\s*(\d+)\s*,\s*(?:Lost|Hilang)\s*=\s*(\d+)", out, re.IGNORECASE)
            unix = re.search(r"(\d+)\s+packets? transmitted\s*,\s*(\d+)\s+(?:packets? )?received", out, re.IGNORECASE)
            if win:
                sent, received = int(win[1]), int(win[2])
            elif unix:
                sent, received = int(unix[1]), int(unix[2])
            else:
                sent, received = PING_N, len(times)
            if len(times) >= 2:
                return {"latency_ms": sum(times)/len(times),
                        "jitter_ms": sum(abs(b-a) for a,b in zip(times,times[1:]))/(len(times)-1),
                        "packet_loss_percent": 100.0 * max(0, sent-received)/max(1,sent),
                        "method": "ICMP", "error": ""}
            return {"latency_ms": None, "jitter_ms": None,
                    "packet_loss_percent": 100.0 * max(0,sent-received)/max(1,sent),
                    "method": "ICMP", "error": "Balasan ping kurang dari 2; jitter tidak dapat dihitung."}
        except (OSError, subprocess.TimeoutExpired):
            pass
    # Tidak memalsukan packet loss ICMP. Hasil gagal koneksi TCP diberi label tersendiri.
    times: list[float] = []
    failures = 0
    for _ in range(PING_N):
        start = time.perf_counter()
        try:
            with socket.create_connection((TARGET, 443), timeout=0.65):
                times.append((time.perf_counter() - start)*1000)
        except OSError:
            failures += 1
    return {
        "latency_ms": sum(times)/len(times) if times else None,
        "jitter_ms": sum(abs(b-a) for a,b in zip(times,times[1:]))/(len(times)-1) if len(times)>=2 else None,
        "packet_loss_percent": 100.0*failures/PING_N,
        "method": "TCP (indikatif)",
        "error": "Cadangan TCP digunakan; persentase kegagalan koneksi bukan packet loss ICMP.",
    }


class Monitor:
    def __init__(self, chosen: str | None):
        self.nic = chosen
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active = False
        self.session_id = ""
        self.rows: list[dict] = []
        self.last_error = ""

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.active:
                return True, "Pengukuran sudah aktif."
            found = wifi_adapters()
            if self.nic:
                if self.nic not in psutil.net_if_stats() or not psutil.net_if_stats()[self.nic].isup:
                    return False, f"Adaptor {self.nic} tidak aktif/tersedia."
            else:
                if not found:
                    return False, "Tidak ditemukan adaptor WiFi aktif. Hubungkan laptop ke WiFi dan ulangi."
                self.nic = found[0]
            self.session_id = uuid.uuid4().hex[:12]
            self.rows = []
            self.last_error = ""
            self.stop_event = threading.Event()
            self.active = True
            self.thread = threading.Thread(target=self._measure, daemon=True, name="LiveStreamQoS-WiFi")
            self.thread.start()
            return True, "Monitoring dimulai."

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.active = False

    def _measure(self):
        try:
            while not self.stop_event.is_set():
                start_t = time.monotonic()
                start_wall = datetime.now().isoformat(timespec="milliseconds")
                before = psutil.net_io_counters(pernic=True).get(self.nic)
                probe = ping_probe()
                left = PERIOD - (time.monotonic()-start_t)
                if left > 0 and self.stop_event.wait(left):
                    break
                if self.stop_event.is_set():
                    break
                seconds = time.monotonic()-start_t
                after = psutil.net_io_counters(pernic=True).get(self.nic)
                up = None
                if before and after and after.bytes_sent >= before.bytes_sent and seconds > 0:
                    up = (after.bytes_sent-before.bytes_sent)*8 / seconds / 1_000_000
                row = {
                    "session_id": self.session_id,
                    "interval_start": start_wall,
                    "interval_end": datetime.now().isoformat(timespec="milliseconds"),
                    "interval_seconds": round(seconds, 3),
                    "interface_name": self.nic,
                    "latency_ms": probe["latency_ms"],
                    "jitter_ms": probe["jitter_ms"],
                    "packet_loss_percent": probe["packet_loss_percent"],
                    "throughput_mbps": up,
                    "method": probe["method"],
                    "note": probe["error"],
                }
                row["read_status"] = "ok" if all(isinstance(row[k], (int,float)) for k in VALID_KEYS) and 4<=seconds<=6 else "failed"
                with self.lock:
                    self.rows = (self.rows + [row])[-300:]
        except Exception as exc:
            with self.lock:
                self.last_error = f"Pengukuran terhenti: {type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.active = False

    def status(self) -> dict:
        with self.lock:
            return {"active": self.active, "interface": self.nic,
                    "session_id": self.session_id, "rows": self.rows[-50:],
                    "last_error": self.last_error,
                    "interval_seconds": PERIOD,
                    "wifi_candidates": wifi_adapters()}


MONITOR: Monitor


class Handler(BaseHTTPRequestHandler):
    def _origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if allowed_origin(origin) else None

    def _headers(self, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Origin")
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send(self, data: dict, code=200):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(code)
        self.wfile.write(payload)

    def _auth(self) -> bool:
        origin = self.headers.get("Origin")
        if not allowed_origin(origin):
            self._send({"ok": False, "error": "Akses dari origin ini tidak diizinkan."}, 403)
            return False
        return True

    def do_OPTIONS(self):
        if not self._auth():
            return
        self._headers(204)

    def do_GET(self):
        if not self._auth():
            return
        if self.path == "/status":
            self._send({"ok": True, **MONITOR.status()})
        else:
            self._send({"ok": False, "error": "Path tidak tersedia."}, 404)

    def do_POST(self):
        if not self._auth():
            return
        if self.path == "/start":
            ok, message = MONITOR.start()
            self._send({"ok": ok, "message": message, **MONITOR.status()}, 200 if ok else 409)
        elif self.path == "/stop":
            MONITOR.stop()
            self._send({"ok": True, "message": "Monitoring dihentikan.", **MONITOR.status()})
        else:
            self._send({"ok": False, "error": "Path tidak tersedia."}, 404)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="LiveStreamQoS: agen monitoring WiFi lokal")
    parser.add_argument("--interface", help="Nama antarmuka WiFi spesifik (opsional)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--origin", action="append", default=[], help="Origin HTTPS Streamlit tambahan, jika memakai subdomain berbeda")
    args = parser.parse_args()
    for origin in args.origin:
        parsed = urlsplit(origin)
        if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".streamlit.app"):
            sys.exit("Origin tambahan harus berupa HTTPS .streamlit.app yang sah.")
        APP_ORIGINS.add(origin.rstrip("/"))
    global MONITOR
    MONITOR = Monitor(args.interface)
    print("LiveStreamQoS - Agen Pengukuran WiFi")
    print(f"Alamat lokal : http://127.0.0.1:{args.port}")
    print("Adaptor terdeteksi:", ", ".join(wifi_adapters()) or "tidak ditemukan")
    print("Biarkan jendela ini terbuka selama monitoring online berlangsung.")
    try:
        ThreadingHTTPServer((HOST,args.port),Handler).serve_forever()
    except KeyboardInterrupt:
        MONITOR.stop()
        print("Agen dihentikan.")
    except OSError as exc:
        sys.exit(f"Tidak dapat membuka port {args.port}: {exc}")

if __name__ == "__main__":
    main()
