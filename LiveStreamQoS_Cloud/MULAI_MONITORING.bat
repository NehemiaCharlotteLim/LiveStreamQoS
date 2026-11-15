@echo off
chcp 65001 >nul
cd /d "%~dp0"
title LiveStreamQoS - Sensor Wi-Fi Lokal
echo ===============================================
echo     LiveStreamQoS - Mulai Monitoring Wi-Fi
echo ===============================================
echo.
python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 (
  echo Python 3.10 atau lebih baru belum tersedia.
  echo Instal dari https://www.python.org/downloads/windows/
  echo Lalu klik dua kali file ini lagi.
  pause
  exit /b 1
)
python -c "import psutil" >nul 2>&1
if errorlevel 1 (
  echo Memasang kebutuhan sensor psutil satu kali...
  python -m pip install --disable-pip-version-check psutil
  if errorlevel 1 (
    echo Gagal memasang psutil. Periksa koneksi internet.
    pause
    exit /b 1
  )
)
echo Mengukur Wi-Fi laptop dan membuka website secara otomatis...
echo JANGAN TUTUP jendela ini selama monitoring.
echo.
python wifi_sensor_local.py
echo.
echo Sensor sudah berhenti. Tekan tombol untuk menutup jendela.
pause
