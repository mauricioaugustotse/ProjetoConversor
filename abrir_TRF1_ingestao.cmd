@echo off
rem Abre a GUI de ingestao de boletins do TRF1: detecta PDFs novos na pasta, processa
rem (PDF->CSV com noticias via Gemini), importa no Notion (base 'trf1') e reindexa a RAG.
set "PYW=C:\Users\mauri\AppData\Local\Programs\Python\Python313\pythonw.exe"
if exist "%PYW%" (
  start "" "%PYW%" "%~dp0TRF1_ingestao_gui.py"
) else (
  start "" pythonw "%~dp0TRF1_ingestao_gui.py"
  if errorlevel 1 start "" pyw "%~dp0TRF1_ingestao_gui.py"
)
