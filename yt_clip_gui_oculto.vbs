' Abre a GUI "Recortar YouTube" sem janela de console.
' A GUI (WinForms) tem caixa de log propria e grava log de sessao em %TEMP%\yt_clip_gui_*.log.
Option Explicit
Dim shell, base, ps1, cmd
Set shell = CreateObject("WScript.Shell")
base = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
ps1 = base & "yt_clip_gui.ps1"
cmd = "powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -WindowStyle Hidden -File """ & ps1 & """"
shell.CurrentDirectory = base
shell.Run cmd, 0, False
