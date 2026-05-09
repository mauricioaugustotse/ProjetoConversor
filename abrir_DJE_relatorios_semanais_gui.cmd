@echo off
setlocal
cd /d "%~dp0"
py -3 "DJE_relatorios_semanais_gui.py"
if errorlevel 1 (
  echo.
  echo A GUI foi encerrada com erro.
  pause
)
