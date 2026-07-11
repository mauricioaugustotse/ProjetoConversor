@echo off
rem Abre a GUI do Normalizador de PDFs do Notion: carrega as subpaginas de uma
rem pagina-mae, diagnostica o ruido de OCR, gera previa local e aplica a
rem normalizacao in-place (com backup) no Notion.
set "PYW=C:\Users\mauri\AppData\Local\Programs\Python\Python313\pythonw.exe"
if exist "%PYW%" (
  start "" "%PYW%" "%~dp0normalizador_gui.py"
) else (
  start "" pythonw "%~dp0normalizador_gui.py"
  if errorlevel 1 start "" pyw "%~dp0normalizador_gui.py"
)
