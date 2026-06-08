@echo off
rem Lancador da interface do Gerador de Informacao Tecnica + Proposicao (CONLE)
cd /d "%~dp0"
start "" pythonw "%~dp0conle_gerador_gui.pyw"
if errorlevel 1 (
  rem Fallback: usa o py launcher em modo janela
  start "" pyw "%~dp0conle_gerador_gui.pyw"
)
