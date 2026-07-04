# -*- coding: utf-8 -*-
"""Interface gráfica do Conversor Notion → Informação Técnica (CONLE).

Cole a URL (ou ID) de uma página do Notion da Consultoria Legislativa e gere,
com um clique, a Informação Técnica e a minuta de proposição correspondentes,
salvas nas pastas-padrão da casa.
"""
from __future__ import annotations

import datetime
import os
import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# garante import do pacote quando executado por duplo-clique
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conle_conversor import config
from conle_conversor.config import data_extenso
from conle_conversor.pipeline import converter

VERDE = "#3E6F30"
VERDE_ESCURO = "#2c4f22"
CINZA = "#f4f5f2"


class ConversorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CONLE · Conversor Notion → Informação Técnica")
        self.geometry("820x800")
        self.minsize(720, 700)
        self.configure(bg=CINZA)
        ico = Path(__file__).resolve().parent / "icones" / "conle_conversor.ico"
        if ico.exists():
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass
        self._fila: queue.Queue = queue.Queue()
        self._rodando = False
        self._ultimos_caminhos = []

        self._montar_estilos()
        self._montar_widgets()
        self.after(120, self._consumir_fila)

    # ---------------------------------------------------------------- estilo
    def _montar_estilos(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TFrame", background=CINZA)
        st.configure("TLabel", background=CINZA, font=("Segoe UI", 10))
        st.configure("Cab.TLabel", background=CINZA, foreground=VERDE_ESCURO,
                     font=("Segoe UI Semibold", 16))
        st.configure("Sub.TLabel", background=CINZA, foreground="#555",
                     font=("Segoe UI", 9))
        st.configure("Sec.TLabelframe", background=CINZA)
        st.configure("Sec.TLabelframe.Label", background=CINZA, foreground=VERDE_ESCURO,
                     font=("Segoe UI Semibold", 10))
        st.configure("TCheckbutton", background=CINZA, font=("Segoe UI", 10))
        st.configure("Gerar.TButton", font=("Segoe UI Semibold", 11), padding=8)
        st.configure("TButton", font=("Segoe UI", 9), padding=3)

    # --------------------------------------------------------------- widgets
    def _montar_widgets(self):
        topo = ttk.Frame(self, padding=(18, 16, 18, 6))
        topo.pack(fill="x")
        ttk.Label(topo, text="Conversor Notion → Informação Técnica", style="Cab.TLabel").pack(anchor="w")
        ttk.Label(topo, text="Consultoria Legislativa · transpõe a página do Notion em IT + minuta ou parecer de comissão (.docx)",
                  style="Sub.TLabel").pack(anchor="w")

        corpo = ttk.Frame(self, padding=(18, 4, 18, 4))
        corpo.pack(fill="both", expand=True)

        # URL ---------------------------------------------------------------
        ttk.Label(corpo, text="Página do Notion (URL ou ID):", font=("Segoe UI Semibold", 10)).pack(anchor="w")
        self.var_url = tk.StringVar()
        e = ttk.Entry(corpo, textvariable=self.var_url, font=("Consolas", 10))
        e.pack(fill="x", pady=(2, 10))
        e.focus()

        # duas colunas: identificação | saída ------------------------------
        cols = ttk.Frame(corpo)
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1, uniform="c")
        cols.columnconfigure(1, weight=1, uniform="c")

        ident = ttk.Labelframe(cols, text=" Identificação (opcional) ", style="Sec.TLabelframe", padding=10)
        ident.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.var_dep = tk.StringVar()
        self.var_sisconle = tk.StringVar()
        self.var_consultor = tk.StringVar(value=config.CONSULTOR_NOME)
        self.var_vocativo = tk.StringVar(value=config.VOCATIVO_DEFAULT)
        hoje = datetime.date.today()
        self.var_data = tk.StringVar(value=data_extenso(hoje.day, hoje.month, hoje.year))

        self._campo(ident, "Deputado(a) solicitante:", self.var_dep,
                    dica="deixe em branco para usar placeholder")
        self._campo(ident, "Nº SISCONLE:", self.var_sisconle, dica="ex.: 2026-1234")
        ttk.Label(ident, text="Vocativo:").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(ident, textvariable=self.var_vocativo, state="readonly",
                     values=["Senhor(a) Deputado(a),", "Senhora Deputada,", "Senhor Deputado,"]
                     ).pack(fill="x")
        self._campo(ident, "Consultor(a):", self.var_consultor)
        self._campo(ident, "Data do fecho da IT:", self.var_data)

        saida = ttk.Labelframe(cols, text=" Saída ", style="Sec.TLabelframe", padding=10)
        saida.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.var_it = tk.BooleanVar(value=True)
        self.var_prop = tk.BooleanVar(value=True)
        self.var_ia = tk.BooleanVar(value=True)
        ttk.Checkbutton(saida, text="Gerar Informação Técnica", variable=self.var_it).pack(anchor="w")
        ttk.Checkbutton(saida, text="Gerar minuta de proposição", variable=self.var_prop).pack(anchor="w")
        ttk.Checkbutton(saida, text="Harmonizar abertura com IA (OpenAI)", variable=self.var_ia).pack(anchor="w", pady=(0, 6))

        self.var_dir_it = tk.StringVar(value=str(config.OUTPUT_IT_DIR))
        self.var_dir_prop = tk.StringVar(value=str(config.OUTPUT_PROPOSICAO_DIR))
        self.var_dir_parecer = tk.StringVar(value=str(config.OUTPUT_PARECER_DIR))
        self._campo_pasta(saida, "Pasta da IT:", self.var_dir_it)
        self._campo_pasta(saida, "Pasta da minuta:", self.var_dir_prop)
        self._campo_pasta(saida, "Pasta do parecer:", self.var_dir_parecer)
        ttk.Label(saida, text="Página com layout de parecer de comissão é detectada\ne gerada automaticamente (parecer + substitutivo).",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        # botão -------------------------------------------------------------
        acoes = ttk.Frame(corpo)
        acoes.pack(fill="x", pady=(12, 6))
        self.btn = ttk.Button(acoes, text="Gerar documentos", style="Gerar.TButton", command=self._on_gerar)
        self.btn.pack(side="left")
        self.btn_abrir = ttk.Button(acoes, text="Abrir pasta de saída", command=self._abrir_pasta, state="disabled")
        self.btn_abrir.pack(side="left", padx=8)

        # log ---------------------------------------------------------------
        ttk.Label(corpo, text="Progresso:", font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(6, 2))
        self.log = tk.Text(corpo, height=11, font=("Consolas", 9), bg="#1e1e1e", fg="#d6f5c6",
                           relief="flat", wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

        self.barra = ttk.Progressbar(corpo, mode="indeterminate")
        self.barra.pack(fill="x", pady=(6, 0))

    def _campo(self, parent, label, var, dica=None):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=var).pack(fill="x")
        if dica:
            ttk.Label(parent, text=dica, style="Sub.TLabel").pack(anchor="w")

    def _campo_pasta(self, parent, label, var):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        linha = ttk.Frame(parent)
        linha.pack(fill="x")
        ttk.Entry(linha, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(linha, text="…", width=3,
                   command=lambda: self._escolher_pasta(var)).pack(side="left", padx=(4, 0))

    def _escolher_pasta(self, var):
        inicial = var.get() if os.path.isdir(var.get()) else os.path.expanduser("~")
        d = filedialog.askdirectory(initialdir=inicial, title="Escolha a pasta de saída")
        if d:
            var.set(d)

    # ---------------------------------------------------------------- ações
    def _on_gerar(self):
        if self._rodando:
            return
        url = self.var_url.get().strip()
        if not url:
            messagebox.showwarning("Atenção", "Cole a URL (ou ID) da página do Notion.")
            return
        # a validação "selecione ao menos um" fica no pipeline: página-parecer
        # é detectada pela estrutura e gera o parecer mesmo sem checkbox marcado

        overrides = {
            "deputado_nome": self.var_dep.get().strip(),
            "sisconle": self.var_sisconle.get().strip(),
            "consultor": self.var_consultor.get().strip() or config.CONSULTOR_NOME,
            "vocativo": self.var_vocativo.get().strip() or config.VOCATIVO_DEFAULT,
            "data_fecho_it": self.var_data.get().strip(),
        }
        params = dict(
            url=url,
            usar_ia=self.var_ia.get(),
            gerar_it=self.var_it.get(),
            gerar_proposicao=self.var_prop.get(),
            overrides=overrides,
            out_it_dir=Path(self.var_dir_it.get()),
            out_prop_dir=Path(self.var_dir_prop.get()),
            out_parecer_dir=Path(self.var_dir_parecer.get()),
        )

        self._rodando = True
        self.btn.config(state="disabled", text="Gerando…")
        self.btn_abrir.config(state="disabled")
        self._limpar_log()
        self.barra.start(12)
        threading.Thread(target=self._worker, kwargs=params, daemon=True).start()

    def _worker(self, **params):
        try:
            res = converter(progress=lambda m: self._fila.put(("log", m)), **params)
            self._fila.put(("ok", res))
        except Exception as exc:
            self._fila.put(("erro", f"{exc}\n\n{traceback.format_exc()}"))

    def _consumir_fila(self):
        try:
            while True:
                tipo, dado = self._fila.get_nowait()
                if tipo == "log":
                    self._append_log(dado)
                elif tipo == "ok":
                    self._fim_ok(dado)
                elif tipo == "erro":
                    self._fim_erro(dado)
        except queue.Empty:
            pass
        self.after(120, self._consumir_fila)

    def _fim_ok(self, res):
        self._rodando = False
        self.barra.stop()
        self.btn.config(state="normal", text="Gerar documentos")
        self._ultimos_caminhos = [str(c) for c in res.caminhos]
        self._append_log("")
        self._append_log(f"✓ {len(res.caminhos)} documento(s) gerado(s) — tipo: {res.tipo_sigla}.")
        for c in res.caminhos:
            self._append_log(f"   • {c}")
        if res.avisos:
            for a in res.avisos:
                self._append_log(f"   ⚠ {a}")
        if self._ultimos_caminhos:
            self.btn_abrir.config(state="normal")
        messagebox.showinfo("Concluído", f"{len(res.caminhos)} documento(s) gerado(s) com sucesso.")

    def _fim_erro(self, msg):
        self._rodando = False
        self.barra.stop()
        self.btn.config(state="normal", text="Gerar documentos")
        self._append_log("")
        self._append_log("✗ ERRO:")
        self._append_log(msg)
        messagebox.showerror("Erro", msg.splitlines()[0] if msg else "Falha na conversão.")

    def _abrir_pasta(self):
        if not self._ultimos_caminhos:
            return
        pasta = os.path.dirname(self._ultimos_caminhos[0])
        try:
            os.startfile(pasta)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(pasta)

    # ------------------------------------------------------------------ log
    def _limpar_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _append_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    ConversorGUI().mainloop()
