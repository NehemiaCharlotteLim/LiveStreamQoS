@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)
%PY% -c "import psutil" >nul 2>&1
if errorlevel 1 (
    echo Menginstal pustaka pengukuran WiFi...
    %PY% -m pip install psutil
    if errorlevel 1 goto failed
)
echo Menjalankan agen WiFi LiveStreamQoS di laptop ini...
%PY% wifi_agent.py
:failed
echo.
echo Jika muncul masalah, kirimkan teks error dari jendela ini.
pause
