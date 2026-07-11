@echo off
rem Abre a GUI dos Relatorios Semanais do DJE sem janela de console.
set "PYW=C:\Users\mauri\AppData\Local\Programs\Python\Python313\pythonw.exe"
if exist "%PYW%" (
  start "" /d "%~dp0" "%PYW%" "%~dp0DJE_relatorios_semanais_gui.py"
) else (
  start "" /d "%~dp0" pythonw "%~dp0DJE_relatorios_semanais_gui.py"
  if errorlevel 1 start "" /d "%~dp0" pyw "%~dp0DJE_relatorios_semanais_gui.py"
)
