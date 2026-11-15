@echo off
setlocal
cd /d "%~dp0"
title LiveStreamQoS - Monitoring Wi-Fi
where python >nul 2>nul
if errorlevel 1 (
  echo Python tidak ditemukan. Pasang Python 3.11 dan aktifkan PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Menyiapkan lingkungan aplikasi untuk penggunaan pertama...
  python -m venv .venv
  if errorlevel 1 goto :fail
)
if not exist ".venv\ready.txt" (
  echo Memasang library aplikasi. Ini dilakukan sekali saja...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  echo ready> ".venv\ready.txt"
)
echo Menjalankan LiveStreamQoS. Buka alamat http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo Gagal menjalankan aplikasi. Baca pesan di atas.
pause
exit /b 1
