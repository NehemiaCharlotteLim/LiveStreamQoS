"""Agen WiFi lokal. Jalankan di laptop yang mengakses aplikasi Streamlit.

Ping ICMP mengukur RTT/kehilangan respons ke target. psutil membaca bytes
received adaptor WiFi (bukan kapasitas internet atau stream video).
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import platform
import re
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlparse


def args_parse():
    ap = argparse.ArgumentParser(description="Pengumpul QoS WiFi untuk LiveStreamQoS")
    ap.add_argument("--origin", required=True, help="URL Streamlit Cloud Anda, mis. https://livestreamqos.streamlit.app")
    ap.add_argument("--target", default="1.1.1.1", help="Host IP target ping; default 1.1.1.1")
    ap.add_argument("--interface", default="", help="Nama adaptor WiFi (opsional), mis. Wi-Fi")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--csv-output", default="pengukuran_wifi.csv", help="Simpan data valid ke CSV di laptop")
    return ap.parse_args()


def get_rtt(target: str) -> float | None:
    if platform.system().lower().startswith("win"):
        cmd = ["ping", "-n", "1", "-w", "750", target]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", target]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2,
                              errors="replace", check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # Contoh hasil Windows: time<1ms, time=3ms; Linux: time=3.41 ms.
    match = re.search(r"(?:time|waktu)\s*([=<])\s*(\d+(?:[.,]\d+)?)\s*ms", proc.stdout, re.I)
    if not match:
        return None
    rtt = float(match.group(2).replace(",", "."))
    return 0.5 if match.group(1) == "<" and rtt <= 1 else rtt


def choose_wifi(nic_stats: dict, io: dict, preferred: str) -> str | None:
    keys = [k for k in io if k in nic_stats and nic_stats[k].isup]
    if preferred:
        return preferred if preferred in keys else None
    # Nama umum Windows/ Linux. Untuk nama khusus gunakan --interface.
    for hint in ("wi-fi", "wifi", "wlan", "wireless", "wl"):
        for key in keys:
            if hint in key.lower():
                return key
    return None


def main():
    opt = args_parse()
    url = urlparse(opt.origin)
    if url.scheme != "https" or not url.hostname:
        raise SystemExit("--origin harus URL HTTPS aplikasi Streamlit, mis. https://livestreamqos.streamlit.app")
    allowed_origin = f"{url.scheme}://{url.netloc}".lower()
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit("Library psutil belum ada. Jalankan: py -m pip install psutil") from exc

    lock = threading.Lock()
    history = deque(maxlen=12)
    state = {"last_update": None, "error": "Menunggu pengukuran WiFi", "interface": None}
    interval_seconds = 5
    csv_path = Path(opt.csv_output).resolve()
    csv_columns = ["latency_ms", "jitter_ms", "packet_loss_percent", "throughput_mbps"]

    def append_csv(sample):
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(sample)

    def run_collector():
        prev_recv = None
        prev_clock = None
        while True:
            started = time.monotonic()
            io = psutil.net_io_counters(pernic=True)
            nic_stats = psutil.net_if_stats()
            iface = choose_wifi(nic_stats, io, opt.interface)
            if iface is None:
                with lock:
                    state["error"] = "Adaptor WiFi aktif tidak ditemukan. Lihat nama adaptor dengan --interface."
                    state["interface"] = None
                time.sleep(interval_seconds)
                continue
            now_bytes = io[iface].bytes_recv
            clock = time.monotonic()
            if prev_recv is None or prev_clock is None or state["interface"] != iface or now_bytes < prev_recv:
                throughput = None
            else:
                elapsed = max(clock - prev_clock, 1e-6)
                throughput = max(0.0, (now_bytes - prev_recv) * 8 / elapsed / 1_000_000)
            prev_recv, prev_clock = now_bytes, clock
            rtts = [get_rtt(opt.target) for _ in range(4)]
            valid = [r for r in rtts if r is not None]
            if valid and throughput is not None:
                latency = sum(valid) / len(valid)
                jitter = (sum(abs(valid[i] - valid[i - 1]) for i in range(1, len(valid))) / (len(valid) - 1)) if len(valid) >= 2 else None
                if jitter is not None:
                    sample = {
                        "latency_ms": round(latency, 4),
                        "jitter_ms": round(jitter, 4),
                        "packet_loss_percent": round(100 * (4 - len(valid)) / 4, 4),
                        "throughput_mbps": round(throughput, 6),
                    }
                    append_csv(sample)
                    with lock:
                        history.append(sample)
                        state.update({"last_update": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                                      "error": None, "interface": iface})
                else:
                    with lock:
                        state["error"] = "Respons ping terlalu sedikit untuk mengukur jitter; interval ini tidak dihitung."
            else:
                with lock:
                    state["error"] = "Pengukuran valid belum tersedia. Periksa ping, nama WiFi, dan koneksi internet."
                    state["interface"] = iface
            print(f"WiFi={iface} | interval valid={len(history)}/12 | {state['error'] or 'OK'}", flush=True)
            time.sleep(max(0.0, interval_seconds - (time.monotonic() - started)))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _cors(self):
            origin = self.headers.get("Origin", "").lower()
            # Komponen Streamlit dalam iframe terkadang memakai origin 'null'.
            # Endpoint dibatasi loopback dan hanya mengekspor statistik QoS.
            if origin == allowed_origin or origin == "null":
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Cache-Control", "no-store")

        def do_OPTIONS(self):
            origin = self.headers.get("Origin", "").lower()
            if origin not in (allowed_origin, "null"):
                self.send_error(403)
                return
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path != "/qos":
                self.send_error(404)
                return
            origin = self.headers.get("Origin", "").lower()
            # Header Origin kosong diperbolehkan untuk diagnostik localhost saja.
            if origin and origin not in (allowed_origin, "null"):
                self.send_error(403)
                return
            with lock:
                result = {"samples": list(history), **state, "type": "local_wifi"}
            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    worker = threading.Thread(target=run_collector, daemon=True)
    worker.start()
    server = ThreadingHTTPServer(("127.0.0.1", opt.port), Handler)
    print(f"Agen WiFi aktif di http://127.0.0.1:{opt.port}/qos", flush=True)
    print(f"Origin browser yang diizinkan: {allowed_origin}", flush=True)
    print(f"Cadangan CSV: {csv_path}", flush=True)
    print("Buka aplikasi Streamlit dari browser pada laptop yang SAMA.", flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("Agen dihentikan.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
