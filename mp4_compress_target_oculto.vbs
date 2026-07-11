' Abre a GUI "Reduzir MP4 para Tamanho Alvo" sem janela de console.
' Aceita um arquivo como argumento (arrastar e soltar) e repassa via -InputFile.
' A GUI (WinForms) tem log proprio e grava log de sessao em %TEMP%\mp4_compress_gui_*.log.
Option Explicit
Dim shell, base, ps1, cmd
Set shell = CreateObject("WScript.Shell")
base = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
ps1 = base & "mp4_compress_target.ps1"
cmd = "powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -WindowStyle Hidden -File """ & ps1 & """"
If WScript.Arguments.Count > 0 Then
    cmd = cmd & " -InputFile """ & WScript.Arguments(0) & """"
End If
shell.CurrentDirectory = base
shell.Run cmd, 0, False
