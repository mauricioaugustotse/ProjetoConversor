@echo off
rem Lancador da interface do Conversor Notion -> Informacao Tecnica (CONLE)
cd /d "%~dp0"
start "" pythonw "%~dp0conle_conversor_gui.pyw"
if errorlevel 1 (
  rem Fallback: usa o py launcher em modo janela
  start "" pyw "%~dp0conle_conversor_gui.pyw"
)
