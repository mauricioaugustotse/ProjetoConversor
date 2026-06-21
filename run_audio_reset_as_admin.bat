@echo off
setlocal

net session >nul 2>&1
if not "%errorlevel%"=="0" (
    echo Requesting Administrator permission...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""%~dp0reset_audio_non_bluetooth.ps1""'"
    exit /b
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0reset_audio_non_bluetooth.ps1"
pause
