# -*- coding: utf-8 -*-
"""Baixar do Instagram — GUI mínima para baixar vídeos do Instagram (e de qualquer
site suportado pelo yt-dlp) direto para a pasta Downloads do usuário.

Uso normal: atalho da Área de Trabalho (pythonw, sem console).
Debug com console: py insta_baixar_gui.py
"""
from __future__ import annotations

import glob
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading

# Sob pythonw, stdout/stderr são None; bibliotecas que escrevem neles quebrariam.
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

CREATE_NO_WINDOW = 0x08000000
RE_URL = re.compile(r"https?://\S+", re.I)
RE_INSTAGRAM = re.compile(r"https?://(www\.)?instagram\.com/\S+", re.I)
# Combobox -> argumento de --cookies-from-browser do yt-dlp
NAVEGADORES = {"Sem login": None, "Chrome": "chrome", "Edge": "edge", "Firefox": "firefox"}


def pasta_downloads() -> str:
    """Pasta Downloads real do usuário (registro do Windows; fallback ~/Downloads)."""
    try:
        import winreg
        chave = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, chave) as k:
            valor, _ = winreg.QueryValueEx(k, "{374DE290-123F-4565-9164-39C4925E467B}")
        return os.path.expandvars(valor)
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Downloads")


def achar_ffmpeg() -> str | None:
    """Localiza o ffmpeg mesmo quando o PATH da sessão ainda não foi atualizado
    (instalação recente via winget)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return os.path.dirname(exe)
    padrao = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                          "Microsoft", "WinGet", "Packages",
                          "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe")
    achados = glob.glob(padrao)
    if achados:
        return os.path.dirname(sorted(achados)[-1])
    return None


class _LoggerYtdlp:
    """Roteia as mensagens do yt-dlp para a caixa de log da GUI."""

    def __init__(self, emit):
        self.emit = emit

    def debug(self, msg):
        if not msg.startswith("[debug]"):
            self.emit(msg)

    def info(self, msg):
        self.emit(msg)

    def warning(self, msg):
        self.emit(f"AVISO: {msg}")

    def error(self, msg):
        self.emit(f"ERRO: {msg}")


def baixar(url: str, destino: str | None = None, navegador: str | None = None,
           emit=print, progresso=None) -> list[str]:
    """Baixa `url` para `destino` e retorna os arquivos gravados."""
    import yt_dlp

    destino = destino or pasta_downloads()
    arquivos: list[str] = []   # partes baixadas (podem ser apagadas no merge)
    finais: list[str] = []     # arquivos finais, após pós-processamento

    def hook(d):
        if d.get("status") == "downloading" and progresso:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progresso(d.get("downloaded_bytes", 0) * 100.0 / total)
        elif d.get("status") == "finished":
            if d.get("filename"):
                arquivos.append(d["filename"])
            if progresso:
                progresso(100.0)

    def pp_hook(d):
        if d.get("status") == "finished":
            fp = (d.get("info_dict") or {}).get("filepath")
            if fp and fp not in finais:
                finais.append(fp)

    opts = {
        "outtmpl": os.path.join(destino, "%(title).80s [%(id)s].%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "logger": _LoggerYtdlp(emit),
        "progress_hooks": [hook],
        "postprocessor_hooks": [pp_hook],
        "windowsfilenames": True,
        "retries": 3,
    }
    ffmpeg_dir = achar_ffmpeg()
    if ffmpeg_dir:
        opts["ffmpeg_location"] = ffmpeg_dir
    if navegador:
        opts["cookiesfrombrowser"] = (navegador,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return finais or [a for a in arquivos if os.path.exists(a)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Baixar do Instagram")
        self.geometry("760x420")
        self.minsize(560, 320)
        self._fila: queue.Queue = queue.Queue()
        self._ocupado = False
        self._montar()
        self._auto_colar()
        self.after(100, self._drenar_fila)

    def _montar(self):
        pad = {"padx": 8, "pady": 4}
        topo = ttk.Frame(self)
        topo.pack(fill="x", **pad)
        ttk.Label(topo, text="Link:").pack(side="left")
        self.ent_url = ttk.Entry(topo)
        self.ent_url.pack(side="left", fill="x", expand=True, padx=6)
        self.ent_url.bind("<Return>", lambda _e: self._iniciar())
        ttk.Button(topo, text="Colar", command=self._colar).pack(side="left")

        linha2 = ttk.Frame(self)
        linha2.pack(fill="x", **pad)
        ttk.Label(linha2, text="Login (só se o post for privado):").pack(side="left")
        self.cbo_login = ttk.Combobox(linha2, values=list(NAVEGADORES), width=12,
                                      state="readonly")
        self.cbo_login.set("Sem login")
        self.cbo_login.pack(side="left", padx=6)
        self.btn_baixar = ttk.Button(linha2, text="⬇  Baixar", command=self._iniciar)
        self.btn_baixar.pack(side="left", padx=12)
        ttk.Button(linha2, text="Abrir Downloads",
                   command=lambda: os.startfile(pasta_downloads())).pack(side="left")
        ttk.Button(linha2, text="Atualizar yt-dlp",
                   command=self._atualizar_ytdlp).pack(side="right")

        self.barra = ttk.Progressbar(self, maximum=100)
        self.barra.pack(fill="x", **pad)

        self.log = ScrolledText(self, height=12, state="disabled",
                                font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", expand=True, **pad)
        self._log(f"Destino dos downloads: {pasta_downloads()}")
        self._log("Cole um link do Instagram e clique em Baixar (ou tecle Enter).")

    # ------------------------------------------------------------------ util
    def _log(self, texto: str):
        self.log.configure(state="normal")
        self.log.insert("end", texto.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _colar(self):
        try:
            texto = self.clipboard_get().strip()
        except tk.TclError:
            return
        m = RE_URL.search(texto)
        if m:
            self.ent_url.delete(0, "end")
            self.ent_url.insert(0, m.group(0))

    def _auto_colar(self):
        try:
            texto = self.clipboard_get().strip()
        except tk.TclError:
            return
        m = RE_INSTAGRAM.search(texto)
        if m:
            self.ent_url.insert(0, m.group(0))
            self._log("Link do Instagram detectado na área de transferência.")

    def _drenar_fila(self):
        try:
            while True:
                tipo, valor = self._fila.get_nowait()
                if tipo == "log":
                    self._log(valor)
                elif tipo == "prog":
                    self.barra["value"] = valor
                elif tipo == "fim":
                    self._ocupado = False
                    self.btn_baixar.configure(state="normal")
                    if valor:  # sucesso -> limpa o campo p/ o próximo link
                        self.ent_url.delete(0, "end")
        except queue.Empty:
            pass
        self.after(100, self._drenar_fila)

    # ------------------------------------------------------------- download
    def _iniciar(self):
        if self._ocupado:
            return
        url = self.ent_url.get().strip()
        if not RE_URL.match(url):
            self._log("ERRO: cole um link válido (https://...).")
            return
        navegador = NAVEGADORES.get(self.cbo_login.get())
        self._ocupado = True
        self.btn_baixar.configure(state="disabled")
        self.barra["value"] = 0
        self._log("")
        self._log(f">>> Baixando: {url}")
        threading.Thread(target=self._trabalho, args=(url, navegador),
                         daemon=True).start()

    def _trabalho(self, url: str, navegador: str | None):
        emit = lambda t: self._fila.put(("log", t))
        prog = lambda p: self._fila.put(("prog", p))
        try:
            arquivos = baixar(url, navegador=navegador, emit=emit, progresso=prog)
            if arquivos:
                for a in arquivos:
                    emit(f"CONCLUÍDO: {os.path.basename(a)}")
                emit(f"{len(arquivos)} arquivo(s) em {pasta_downloads()}")
            else:
                emit("Nada foi baixado (veja as mensagens acima).")
            self._fila.put(("fim", bool(arquivos)))
        except Exception as e:  # noqa: BLE001 - sob pythonw, exceção some sem isso
            texto = str(e)
            emit(f"FALHOU: {texto}")
            baixo = texto.lower()
            if any(s in baixo for s in ("login", "empty media response", "rate-limit",
                                        "restricted", "not available")):
                emit("DICA: o post pode exigir login. Escolha o navegador em que você "
                     "está logado no Instagram (campo \"Login\") e tente de novo. "
                     "Feche o navegador antes, se ele reclamar do cofre de cookies.")
                emit("DICA 2: se o post é público e mesmo assim falhou, clique em "
                     "\"Atualizar yt-dlp\" — o Instagram muda com frequência.")
            self._fila.put(("fim", False))

    # -------------------------------------------------------------- update
    def _atualizar_ytdlp(self):
        if self._ocupado:
            return
        self._ocupado = True
        self.btn_baixar.configure(state="disabled")
        self._log(">>> Atualizando yt-dlp (pip)...")
        threading.Thread(target=self._trabalho_update, daemon=True).start()

    def _trabalho_update(self):
        emit = lambda t: self._fila.put(("log", t))
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"],
                capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
                timeout=300)
            saida = (r.stdout or "").strip().splitlines()
            if saida:
                emit(saida[-1])
            if r.returncode == 0:
                if "yt_dlp" in sys.modules and "Successfully installed" in (r.stdout or ""):
                    emit("Atualizado. Feche e reabra esta janela para usar a versão nova.")
                else:
                    emit("yt-dlp verificado/atualizado.")
            else:
                emit(f"FALHOU (pip retornou {r.returncode}): {(r.stderr or '').strip()[-400:]}")
        except Exception as e:  # noqa: BLE001
            emit(f"FALHOU: {e}")
        self._fila.put(("fim", False))


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
