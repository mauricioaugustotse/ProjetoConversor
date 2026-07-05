# -*- coding: utf-8 -*-
"""Interface gráfica do Atualizador da base 'resoluções' (Resoluções TSE compiladas).

Raspa https://www.tse.jus.br/legislacao/compilada/res, compara com a base Notion
'Resolucoes TSE - RAG consolidado' e: (1) insere no CATÁLOGO as resoluções novas
(1 linha por norma: número, data, ementa, URL); (2) para as resoluções MONITORADAS,
re-baixa o texto compilado e sincroniza dispositivo a dispositivo (alterações,
inclusões e revogações detectadas pelas notas oficiais). Botão de dry-run mostra
as novidades sem gravar.

Fluxo aberto: qualquer resolução do acervo compilado pode entrar nas monitoradas
— informe nº/ano OU cole o link da página no site do TSE; a GUI valida no índice
anual e mostra a ementa antes de confirmar (resolucoes_work/monitoradas.json).
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


class ResolucoesGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSE · Atualizador — Resoluções compiladas (base resoluções)")
        self.geometry("900x680")
        self.minsize(760, 560)
        self.configure(bg=CINZA)
        self._fila: queue.Queue = queue.Queue()
        self._rodando = False
        self._montar_estilos()
        self._montar_widgets()
        self._atualizar_rotulo_monitoradas()
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
        ttk.Label(topo, text="Atualizador da base de resoluções (TSE)",
                  style="Cab.TLabel").pack(anchor="w")
        ttk.Label(topo, text="Legislação compilada do TSE → base 'resoluções' no Notion: "
                             "catálogo completo por norma + texto integral dispositivo a "
                             "dispositivo das resoluções monitoradas.",
                  style="Sub.TLabel").pack(anchor="w")

        opc = ttk.Frame(self, padding=(18, 6, 18, 2))
        opc.pack(fill="x")
        self.var_reindexar = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Reindexar RAG ao final",
                        variable=self.var_reindexar).pack(side="left")
        self.var_completa = tk.BooleanVar(value=False)
        ttk.Checkbutton(opc, text="Varredura completa (1994–hoje)",
                        variable=self.var_completa).pack(side="left", padx=(14, 0))
        self.var_integrais = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Sincronizar textos integrais (monitoradas)",
                        variable=self.var_integrais).pack(side="left", padx=(14, 0))
        ttk.Label(opc, text="   Limite (vazio = todas):").pack(side="left")
        self.var_limite = tk.StringVar()
        ttk.Entry(opc, textvariable=self.var_limite, width=6).pack(side="left")

        mon = ttk.Frame(self, padding=(18, 4, 18, 2))
        mon.pack(fill="x")
        ttk.Label(mon, text="Monitorar resolução (nº/ano ou link do TSE):").pack(side="left")
        self.var_nova_mon = tk.StringVar()
        ttk.Entry(mon, textvariable=self.var_nova_mon, width=46).pack(side="left", padx=(6, 6))
        self.btn_mon = ttk.Button(mon, text="Adicionar às monitoradas",
                                  command=self._on_adicionar_monitorada)
        self.btn_mon.pack(side="left")
        self.lbl_mon = ttk.Label(self, text="", style="Sub.TLabel", padding=(18, 0, 18, 0),
                                 wraplength=840, justify="left")
        self.lbl_mon.pack(fill="x")

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
                elif tipo == "mon_confirmar":
                    self._confirmar_monitorada(*dado)
        except queue.Empty:
            pass
        self.after(120, self._consumir_fila)

    # ------------------------------------------------------------ monitoradas
    def _atualizar_rotulo_monitoradas(self):
        try:
            import resolucoes_updater as U
            mon = U.carregar_monitoradas()
            nums = ", ".join(f"{U.fmt_numero(k)}/{v}" for k, v in sorted(mon.items()))
            self.lbl_mon.configure(text=f"Monitoradas (texto integral): {nums}")
        except Exception:
            self.lbl_mon.configure(text="Monitoradas: (erro ao ler monitoradas.json)")

    def _on_adicionar_monitorada(self):
        import resolucoes_updater as U
        try:
            num, ano = U.parse_ref_resolucao(self.var_nova_mon.get())
        except ValueError as e:
            messagebox.showwarning("Formato inválido", str(e))
            return
        if num in U.carregar_monitoradas():
            messagebox.showinfo("Já monitorada",
                                f"A Resolução {U.fmt_numero(num)}/{ano} já é monitorada.")
            return
        self.btn_mon.state(["disabled"])
        self.lbl_status.configure(text=f"Conferindo a Res. {U.fmt_numero(num)}/{ano} "
                                       "no índice do TSE...")

        def worker():
            try:
                item = U.localizar_no_indice(num, ano,
                                             progress=lambda m: self._fila.put(("log", m)))
                self._fila.put(("mon_confirmar", (num, ano, item)))
            except Exception:
                self._fila.put(("mon_confirmar", (num, ano, None)))

        threading.Thread(target=worker, daemon=True).start()

    def _confirmar_monitorada(self, num: str, ano: str, item):
        import resolucoes_updater as U
        self.btn_mon.state(["!disabled"])
        self.lbl_status.configure(text="Pronto.")
        rotulo = f"Res.-TSE nº {U.fmt_numero(num)}/{ano}"
        if item:
            ementa = item.get("ementa") or "(sem ementa no índice)"
            ok = messagebox.askyesno(
                "Confirmar monitoramento",
                f"{rotulo}\n\nEmenta: {ementa[:400]}\n\n"
                "Adicionar às monitoradas? O texto integral (dispositivo a dispositivo) "
                "será ingerido na próxima atualização.")
        else:
            ok = messagebox.askyesno(
                "Resolução não encontrada no índice",
                f"Não localizei a {rotulo} no índice {ano} do acervo compilado do TSE.\n\n"
                "Confira o número/link. Deseja monitorá-la mesmo assim?")
        if not ok:
            return
        mon = U.carregar_monitoradas()
        mon[num] = ano
        U.salvar_monitoradas(mon)
        self.var_nova_mon.set("")
        self._atualizar_rotulo_monitoradas()
        self._log(f"Monitorada adicionada: {rotulo} — o texto integral será "
                  f"ingerido na próxima atualização (botão 'Atualizar base no Notion').")

    # ----------------------------------------------------------------- ações
    def _limite(self):
        s = (self.var_limite.get() or "").strip()
        return int(s) if s.isdigit() and int(s) > 0 else None

    def _on_dry(self):
        self._iniciar(dry_run=True)

    def _on_atualizar(self):
        if not messagebox.askyesno(
                "Atualizar base",
                "As resoluções novas entram no catálogo e as monitoradas terão o texto "
                "compilado re-sincronizado (dispositivos novos/alterados/revogados), com "
                "enriquecimento via OpenAI.\n\nContinuar?"):
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
        varredura = bool(self.var_completa.get())
        escopo = "tudo" if self.var_integrais.get() else "catalogo"
        limite = self._limite()

        def worker():
            try:
                import resolucoes_updater
                resumo = resolucoes_updater.executar(
                    dry_run=dry_run, escopo=escopo, limite=limite, reindexar=reindexar,
                    varredura_completa=varredura,
                    progress=lambda m: self._fila.put(("log", m)))
                self._fila.put(("fim", (dry_run, resumo)))
            except Exception:
                self._fila.put(("erro", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------- fim
    def _fim(self, dado):
        dry_run, resumo = dado
        self._parar()
        cat = resumo.get("catalogo") or {}
        ints = resumo.get("integrais") or []
        mudancas = sum(r.get("novos", 0) + r.get("mudados", 0) + r.get("sumidos", 0)
                       for r in ints)
        if dry_run:
            self.lbl_status.configure(
                text=f"Dry-run: {cat.get('novas', 0)} resolução(ões) nova(s), "
                     f"{mudancas} dispositivo(s) com diferença.")
            messagebox.showinfo(
                "Verificação concluída",
                f"Resoluções novas no site: {cat.get('novas', 0)}\n"
                f"Dispositivos novos/alterados/sumidos nas monitoradas: {mudancas}\n\n"
                f"Relatório: {cat.get('csv', '')}")
        else:
            criadas = cat.get("criadas", 0) + sum(r.get("criadas", 0) for r in ints)
            atualizadas = sum(r.get("atualizadas", 0) for r in ints)
            self.lbl_status.configure(
                text=f"Concluído: {criadas} criada(s), {atualizadas} atualizada(s).")
            messagebox.showinfo(
                "Atualização concluída",
                f"Catálogo — novas: {cat.get('novas', 0)}, criadas: {cat.get('criadas', 0)}\n"
                f"Textos integrais — criadas: {sum(r.get('criadas', 0) for r in ints)}, "
                f"atualizadas: {atualizadas}, "
                f"arquivadas: {sum(r.get('arquivadas', 0) for r in ints)}")

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
    ResolucoesGUI().mainloop()
