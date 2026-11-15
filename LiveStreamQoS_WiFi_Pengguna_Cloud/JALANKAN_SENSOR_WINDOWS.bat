@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =========================================
echo    LiveStreamQoS - Sensor Wi-Fi Windows
 echo =========================================
where python >nul 2>&1
if errorlevel 1 (
  echo Python tidak ditemukan. Instal Python 3.11 atau lebih baru terlebih dahulu.
  echo https://www.python.org/downloads/windows/
  pause
  exit /b 1
)
python -m pip install --disable-pip-version-check psutil
if errorlevel 1 (
  echo Gagal memasang psutil. Periksa internet dan Python.
  pause
  exit /b 1
)
python wifi_sensor_local.py
pause
