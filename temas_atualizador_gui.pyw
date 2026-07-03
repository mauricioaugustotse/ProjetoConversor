# -*- coding: utf-8 -*-
"""Interface gráfica do Atualizador da base 'temas' (Jurisprudência TSE por assunto).

Baixa os 22 ramos de https://temasselecionados.tse.jus.br/, compara com a base
Notion 'temas' e insere os julgados NOVOS (enriquecidos via OpenAI, relator
normalizado, incluir_no_rag=True). Botão de dry-run mostra as novidades sem gravar.
"""
from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERDE = "#3E6F30"
VERDE_ESCURO = "#2c4f22"
CINZA = "#f4f5f2"


class TemasGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSE · Atualizador — Jurisprudência por assunto (base temas)")
        self.geometry("860x640")
        self.minsize(720, 540)
        self.configure(bg=CINZA)
        self._fila: queue.Queue = queue.Queue()
        self._rodando = False
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
        st.configure("Sub.TLabel", background=CINZA, foreground="#555", font=("Segoe UI", 9))
        st.configure("TCheckbutton", background=CINZA, font=("Segoe UI", 10))
        st.configure("Acao.TButton", font=("Segoe UI Semibold", 11), padding=8)
        st.configure("TButton", font=("Segoe UI", 9), padding=4)

    # --------------------------------------------------------------- widgets
    def _montar_widgets(self):
        topo = ttk.Frame(self, padding=(18, 16, 18, 4))
        topo.pack(fill="x")
        ttk.Label(topo, text="Atualizador da base de temas (TSE)", style="Cab.TLabel").pack(anchor="w")
        ttk.Label(topo, text="Site 'Temas Selecionados' → base 'temas' no Notion: baixa os 22 ramos, "
                             "detecta julgados novos, enriquece via IA e insere.",
                  style="Sub.TLabel").pack(anchor="w")

        opc = ttk.Frame(self, padding=(18, 6, 18, 2))
        opc.pack(fill="x")
        self.var_reindexar = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Reindexar RAG ao final da atualização",
                        variable=self.var_reindexar).pack(side="left")
        ttk.Label(opc, text="   Limite de inserções (vazio = todos):").pack(side="left")
        self.var_limite = tk.StringVar()
        ttk.Entry(opc, textvariable=self.var_limite, width=6).pack(side="left")

        botoes = ttk.Frame(self, padding=(18, 8, 18, 4))
        botoes.pack(fill="x")
        self.btn_dry = ttk.Button(botoes, text="Verificar novidades (dry-run)",
                                  style="Acao.TButton", command=self._on_dry)
        self.btn_dry.pack(side="left", padx=(0, 10))
        self.btn_run = ttk.Button(botoes, text="Atualizar base no Notion",
                                  style="Acao.TButton", command=self._on_atualizar)
        self.btn_run.pack(side="left")
        self.prog = ttk.Progressbar(botoes, mode="indeterminate", length=200)
        self.prog.pack(side="right")

        corpo = ttk.Frame(self, padding=(18, 4, 18, 8))
        corpo.pack(fill="both", expand=True)
        self.txt = tk.Text(corpo, bg="#1e1e1e", fg="#d8d8d8", insertbackground="#d8d8d8",
                           font=("Consolas", 9), wrap="word", state="disabled")
        scr = ttk.Scrollbar(corpo, command=self.txt.yview)
        self.txt.configure(yscrollcommand=scr.set)
        self.txt.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        rodape = ttk.Frame(self, padding=(18, 0, 18, 12))
        rodape.pack(fill="x")
        self.lbl_status = ttk.Label(rodape, text="Pronto.", style="Sub.TLabel")
        self.lbl_status.pack(side="left")

    # ------------------------------------------------------------------ fila
    def _log(self, msg: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _consumir_fila(self):
        try:
            while True:
                tipo, dado = self._fila.get_nowait()
                if tipo == "log":
                    self._log(str(dado))
                elif tipo == "fim":
                    self._fim(dado)
                elif tipo == "erro":
                    self._erro(dado)
        except queue.Empty:
            pass
        self.after(120, self._consumir_fila)

    # ----------------------------------------------------------------- ações
    def _limite(self):
        s = (self.var_limite.get() or "").strip()
        return int(s) if s.isdigit() and int(s) > 0 else None

    def _on_dry(self):
        self._iniciar(dry_run=True)

    def _on_atualizar(self):
        if not messagebox.askyesno(
                "Atualizar base",
                "Os julgados novos serão enriquecidos via OpenAI e INSERIDOS na base "
                "'temas' do Notion.\n\nContinuar?"):
            return
        self._iniciar(dry_run=False)

    def _iniciar(self, dry_run: bool):
        if self._rodando:
            return
        self._rodando = True
        self.btn_dry.state(["disabled"])
        self.btn_run.state(["disabled"])
        self.prog.start(60)
        self.lbl_status.configure(text="Verificando o site do TSE..." if dry_run
                                  else "Atualizando a base no Notion...")
        self._log("=" * 70)
        reindexar = bool(self.var_reindexar.get()) and not dry_run
        limite = self._limite()

        def worker():
            try:
                import temas_updater
                resumo = temas_updater.executar(
                    dry_run=dry_run, limite=limite, reindexar=reindexar,
                    progress=lambda m: self._fila.put(("log", m)))
                self._fila.put(("fim", (dry_run, resumo)))
            except Exception:
                self._fila.put(("erro", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------- fim
    def _fim(self, dado):
        dry_run, resumo = dado
        self._parar()
        novos, criadas = resumo.get("novos", 0), resumo.get("criadas", 0)
        if dry_run:
            self.lbl_status.configure(text=f"Dry-run: {novos} novidade(s) detectada(s).")
            messagebox.showinfo("Verificação concluída",
                                f"Julgados no site: {resumo.get('site', '?')}\n"
                                f"Na base Notion: {resumo.get('base', '?')}\n"
                                f"NOVOS detectados: {novos}\n"
                                f"Incompletos: {resumo.get('incompletos', 0)}\n\n"
                                f"Relatório: {resumo.get('csv', '')}")
        else:
            self.lbl_status.configure(text=f"Atualização concluída: {criadas} página(s) criada(s).")
            messagebox.showinfo("Atualização concluída",
                                f"Novos detectados: {novos}\nPáginas criadas: {criadas}")

    def _erro(self, tb: str):
        self._parar()
        self._log(tb)
        self.lbl_status.configure(text="Erro — veja o log.")
        messagebox.showerror("Erro", tb.splitlines()[-1] if tb.strip() else "Erro desconhecido")

    def _parar(self):
        self._rodando = False
        self.prog.stop()
        self.btn_dry.state(["!disabled"])
        self.btn_run.state(["!disabled"])


if __name__ == "__main__":
    TemasGUI().mainloop()
