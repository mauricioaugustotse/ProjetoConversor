@echo off
rem Abre o Atualizador da base 'temas' (Jurisprudencia TSE por assunto) sem console.
start "" pythonw "%~dp0temas_atualizador_gui.pyw"
if errorlevel 1 start "" pyw "%~dp0temas_atualizador_gui.pyw"
