# -*- coding: utf-8 -*-
"""Interface gráfica do Gerador de Informação Técnica + Proposição (CONLE).

Cole a DEMANDA do parlamentar e a URL de uma página EM BRANCO do Notion; o app
pesquisa as bases internas (RAG), a API da Câmara e a web (Gemini) e grava a IT +
minuta na página, na anatomia que o Conversor (outro app) transforma em .docx.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conle_gerador import config_gerador as cfg
from conle_gerador import gerador, notion_rag

VERDE = "#3E6F30"
VERDE_ESCURO = "#2c4f22"
CINZA = "#f4f5f2"


class GeradorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CONLE · Gerador de Informação Técnica + Proposição")
        self.geometry("880x820")
        self.minsize(780, 720)
        self.configure(bg=CINZA)
        self._fila: queue.Queue = queue.Queue()
        self._rodando = False
        self._page_url = ""
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
        st.configure("Cab.TLabel", background=CINZA, foreground=VERDE_ESCURO, font=("Segoe UI Semibold", 16))
        st.configure("Sub.TLabel", background=CINZA, foreground="#555", font=("Segoe UI", 9))
        st.configure("Sec.TLabelframe", background=CINZA)
        st.configure("Sec.TLabelframe.Label", background=CINZA, foreground=VERDE_ESCURO, font=("Segoe UI Semibold", 10))
        st.configure("TCheckbutton", background=CINZA, font=("Segoe UI", 10))
        st.configure("Gerar.TButton", font=("Segoe UI Semibold", 11), padding=8)
        st.configure("TButton", font=("Segoe UI", 9), padding=4)

    # --------------------------------------------------------------- widgets
    def _montar_widgets(self):
        topo = ttk.Frame(self, padding=(18, 16, 18, 4))
        topo.pack(fill="x")
        ttk.Label(topo, text="Gerador de Informação Técnica + Proposição", style="Cab.TLabel").pack(anchor="w")
        ttk.Label(topo, text="Da demanda parlamentar à página do Notion — pesquisa bases internas, API da Câmara e web (Gemini).",
                  style="Sub.TLabel").pack(anchor="w")

        corpo = ttk.Frame(self, padding=(18, 4, 18, 4))
        corpo.pack(fill="both", expand=True)

        ttk.Label(corpo, text="Demanda do parlamentar:", font=("Segoe UI Semibold", 10)).pack(anchor="w")
        self.txt_demanda = scrolledtext.ScrolledText(corpo, height=6, font=("Segoe UI", 10), wrap="word")
        self.txt_demanda.pack(fill="x", pady=(2, 8))
        self.txt_demanda.insert("1.0",
            "Ex.: O Senhor deputado solicita, com base no esboço anexo, elaboração de PEC para alterar "
            "a Constituição Federal para promover a paridade de gênero nas chapas majoritárias…")

        ttk.Label(corpo, text="Página EM BRANCO do Notion (URL ou ID):", font=("Segoe UI Semibold", 10)).pack(anchor="w")
        self.var_url = tk.StringVar()
        ttk.Entry(corpo, textvariable=self.var_url, font=("Consolas", 10)).pack(fill="x", pady=(2, 8))

        cols = ttk.Frame(corpo)
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1, uniform="c")
        cols.columnconfigure(1, weight=1, uniform="c")

        # fontes -----------------------------------------------------------
        fontes = ttk.Labelframe(cols, text=" Fontes de pesquisa ", style="Sec.TLabelframe", padding=10)
        fontes.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.var_bases = {}
        ttk.Label(fontes, text="Bases internas (RAG) — descobertas no Notion:", style="Sub.TLabel").pack(anchor="w")
        try:
            reg = cfg.descobrir_bases()
        except Exception:
            reg = cfg.BASES_RAG
        # conhecidas primeiro (na ordem do BASES_RAG), depois as demais
        ordem = [k for k in cfg.BASES_RAG if k in reg] + [k for k in reg if k not in cfg.BASES_RAG]
        for chave in ordem:
            label = reg[chave].get("label") or chave
            v = tk.BooleanVar(value=chave in cfg.BASES_PADRAO)
            ttk.Checkbutton(fontes, text=label, variable=v).pack(anchor="w")
            self.var_bases[chave] = v
        ttk.Separator(fontes).pack(fill="x", pady=6)
        self.var_camara = tk.BooleanVar(value=True)
        self.var_web = tk.BooleanVar(value=True)
        ttk.Checkbutton(fontes, text="API da Câmara (proposições correlatas)", variable=self.var_camara).pack(anchor="w")
        ttk.Checkbutton(fontes, text="Pesquisa web (Gemini, econômica)", variable=self.var_web).pack(anchor="w")

        # opções -----------------------------------------------------------
        opc = ttk.Labelframe(cols, text=" Opções ", style="Sec.TLabelframe", padding=10)
        opc.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(opc, text="Modelo de redação (OpenAI):").pack(anchor="w")
        self.var_modelo = tk.StringVar(value=cfg.MODEL_REDACAO)
        ttk.Combobox(opc, textvariable=self.var_modelo, state="readonly",
                     values=[cfg.MODEL_REDACAO, cfg.MODEL_REDACAO_PRO]).pack(fill="x")
        ttk.Label(opc, text="Recomendado: gpt-5.5. O 'pro' é ~7× mais lento e caro,", style="Sub.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(opc, text="com qualidade equivalente para esta tarefa.", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(opc, text="A pesquisa web usa sempre o Gemini (econômico).", style="Sub.TLabel").pack(anchor="w")
        ttk.Separator(opc).pack(fill="x", pady=8)
        ttk.Button(opc, text="Indexar bases RAG (1ª vez / atualizar)", command=self._on_indexar).pack(fill="x")
        ttk.Label(opc, text="Necessário antes do 1º uso do RAG.", style="Sub.TLabel").pack(anchor="w")

        # ações ------------------------------------------------------------
        acoes = ttk.Frame(corpo)
        acoes.pack(fill="x", pady=(12, 6))
        self.btn = ttk.Button(acoes, text="Gerar e gravar no Notion", style="Gerar.TButton", command=self._on_gerar)
        self.btn.pack(side="left")
        self.btn_abrir = ttk.Button(acoes, text="Abrir página no Notion", command=self._abrir_pagina, state="disabled")
        self.btn_abrir.pack(side="left", padx=8)

        ttk.Label(corpo, text="Progresso:", font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(6, 2))
        self.log = tk.Text(corpo, height=12, font=("Consolas", 9), bg="#1e1e1e", fg="#d6f5c6",
                           relief="flat", wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        self.barra = ttk.Progressbar(corpo, mode="indeterminate")
        self.barra.pack(fill="x", pady=(6, 0))

    # ---------------------------------------------------------------- ações
    def _bases_sel(self):
        return [c for c, v in self.var_bases.items() if v.get()]

    def _on_indexar(self):
        if self._rodando:
            return
        bases = self._bases_sel()
        if not bases:
            messagebox.showwarning("Atenção", "Marque ao menos uma base interna para indexar.")
            return
        self._iniciar()
        threading.Thread(target=self._worker_indexar, args=(bases,), daemon=True).start()

    def _on_gerar(self):
        if self._rodando:
            return
        demanda = self.txt_demanda.get("1.0", "end").strip()
        url = self.var_url.get().strip()
        if not demanda or demanda.startswith("Ex.:"):
            messagebox.showwarning("Atenção", "Descreva a demanda do parlamentar.")
            return
        if not url:
            messagebox.showwarning("Atenção", "Cole a URL (ou ID) da página EM BRANCO do Notion.")
            return
        params = dict(
            demanda=demanda, page_url=url,
            usar_rag=bool(self._bases_sel()), usar_camara=self.var_camara.get(),
            usar_web=self.var_web.get(), bases_rag=self._bases_sel(),
            model=self.var_modelo.get(),
        )
        self._iniciar()
        threading.Thread(target=self._worker_gerar, kwargs=params, daemon=True).start()

    def _iniciar(self):
        self._rodando = True
        self.btn.config(state="disabled", text="Processando…")
        self.btn_abrir.config(state="disabled")
        self._limpar_log()
        self.barra.start(12)

    def _worker_indexar(self, bases):
        try:
            r = notion_rag.indexar(bases, progress=lambda m: self._fila.put(("log", m)))
            self._fila.put(("ok_idx", r))
        except Exception as exc:
            self._fila.put(("erro", f"{exc}\n\n{traceback.format_exc()}"))

    def _worker_gerar(self, **params):
        try:
            res = gerador.gerar(progress=lambda m: self._fila.put(("log", m)), **params)
            self._fila.put(("ok_gen", res))
        except Exception as exc:
            self._fila.put(("erro", f"{exc}\n\n{traceback.format_exc()}"))

    def _consumir_fila(self):
        try:
            while True:
                tipo, dado = self._fila.get_nowait()
                if tipo == "log":
                    self._append_log(dado)
                elif tipo == "ok_gen":
                    self._fim_gerar(dado)
                elif tipo == "ok_idx":
                    self._fim_indexar(dado)
                elif tipo == "erro":
                    self._fim_erro(dado)
        except queue.Empty:
            pass
        self.after(120, self._consumir_fila)

    def _parar(self):
        self._rodando = False
        self.barra.stop()
        self.btn.config(state="normal", text="Gerar e gravar no Notion")

    def _fim_gerar(self, res):
        self._parar()
        self._page_url = f"https://www.notion.so/{res.page_id.replace('-', '')}"
        self._append_log("")
        self._append_log(f"✓ Gerado e gravado: {res.tipo_sigla} — {res.n_blocos} blocos.")
        self._append_log(f"   Título: {res.titulo}")
        for a in res.avisos:
            self._append_log(f"   ⚠ {a}")
        if res.fontes_web:
            self._append_log("   Fontes web consultadas:")
            for u in res.fontes_web:
                self._append_log(f"     • {u}")
        self.btn_abrir.config(state="normal")
        messagebox.showinfo("Concluído", f"{res.tipo_sigla} gravado na página do Notion ({res.n_blocos} blocos).")

    def _fim_indexar(self, resumo):
        self._parar()
        self._append_log("")
        self._append_log("✓ Indexação concluída: " + ", ".join(f"{k}={v}" for k, v in resumo.items()))
        messagebox.showinfo("Indexação", "Bases indexadas:\n" + "\n".join(f"{k}: {v} trechos" for k, v in resumo.items()))

    def _fim_erro(self, msg):
        self._parar()
        self._append_log("")
        self._append_log("✗ ERRO:")
        self._append_log(msg)
        messagebox.showerror("Erro", msg.splitlines()[0] if msg else "Falha.")

    def _abrir_pagina(self):
        if self._page_url:
            webbrowser.open(self._page_url)

    # ------------------------------------------------------------------ log
    def _limpar_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _append_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", str(msg) + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    GeradorGUI().mainloop()
