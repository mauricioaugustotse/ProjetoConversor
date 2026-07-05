# -*- coding: utf-8 -*-
"""Interface gráfica do Atualizador da base 'vademecum' (Vademecum - RAG consolidado).

Monitora e sincroniza a base com as fontes oficiais: normas federais compiladas
do Planalto (dispositivo a dispositivo, com status de alteração e norma
alteradora), RICD e Código de Ética pela Câmara (DOCX oficial/LEGIN, com detector
de versão) e questões de ordem novas pela API pública da Câmara. Botão de
dry-run mostra as diferenças sem gravar.

Fluxo aberto: cole o link de qualquer norma compilada do Planalto para
adicioná-la ao monitoramento (vademecum_work/normas_extras.json) — a epígrafe
e a ementa são detectadas automaticamente e o texto integral é ingerido na
atualização seguinte.
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


class VademecumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CONLE · Atualizador — Vademecum (Planalto + Câmara)")
        self.geometry("920x720")
        self.minsize(760, 560)
        self.configure(bg=CINZA)
        self._fila: queue.Queue = queue.Queue()
        self._rodando = False
        self._montar_estilos()
        self._montar_widgets()
        self._atualizar_rotulo_extras()
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
        ttk.Label(topo, text="Atualizador do Vademécum (fontes oficiais)",
                  style="Cab.TLabel").pack(anchor="w")
        ttk.Label(topo, text="Planalto (normas compiladas) + Câmara (RICD, Código de Ética e "
                             "questões de ordem) → base 'Vademecum - RAG consolidado' no Notion. "
                             "Diff dispositivo a dispositivo; o texto oficial é autoritativo.",
                  style="Sub.TLabel").pack(anchor="w")

        opc = ttk.Frame(self, padding=(18, 6, 18, 2))
        opc.pack(fill="x")
        self.var_planalto = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Planalto (normas)", variable=self.var_planalto).pack(side="left")
        self.var_camara = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Câmara (RICD + Ética)",
                        variable=self.var_camara).pack(side="left", padx=(12, 0))
        self.var_qordem = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Questões de ordem",
                        variable=self.var_qordem).pack(side="left", padx=(12, 0))
        self.var_reindexar = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc, text="Reindexar RAG ao final",
                        variable=self.var_reindexar).pack(side="left", padx=(12, 0))

        opc2 = ttk.Frame(self, padding=(18, 2, 18, 2))
        opc2.pack(fill="x")
        self.var_refresh = tk.BooleanVar(value=True)
        ttk.Checkbutton(opc2, text="Atualizar retrato da base antes (refresh do dump — recomendado)",
                        variable=self.var_refresh).pack(side="left")
        ttk.Label(opc2, text="   Normas específicas (norma_id, vazio = todas):").pack(side="left")
        self.var_normas = tk.StringVar()
        ttk.Entry(opc2, textvariable=self.var_normas, width=34).pack(side="left")

        add = ttk.Frame(self, padding=(18, 6, 18, 2))
        add.pack(fill="x")
        ttk.Label(add, text="Nova norma (link do Planalto):").pack(side="left")
        self.var_url_nova = tk.StringVar()
        ttk.Entry(add, textvariable=self.var_url_nova, width=46).pack(side="left", padx=(6, 8))
        ttk.Label(add, text="Nome popular:").pack(side="left")
        self.var_nome_pop = tk.StringVar()
        ttk.Entry(add, textvariable=self.var_nome_pop, width=18).pack(side="left", padx=(4, 8))
        self.btn_add = ttk.Button(add, text="Adicionar norma",
                                  command=self._on_adicionar_norma)
        self.btn_add.pack(side="left")
        self.lbl_extras = ttk.Label(self, text="", style="Sub.TLabel",
                                    padding=(18, 0, 18, 0), wraplength=860, justify="left")
        self.lbl_extras.pack(fill="x")

        botoes = ttk.Frame(self, padding=(18, 8, 18, 4))
        botoes.pack(fill="x")
        self.btn_dry = ttk.Button(botoes, text="Verificar diferenças (dry-run)",
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
                elif tipo == "norma_detectada":
                    self._confirmar_norma(dado)
                elif tipo == "norma_erro":
                    self.btn_add.state(["!disabled"])
                    self.lbl_status.configure(text="Pronto.")
                    messagebox.showerror("Não foi possível detectar a norma", str(dado))
        except queue.Empty:
            pass
        self.after(120, self._consumir_fila)

    # ------------------------------------------------------------ normas extras
    def _atualizar_rotulo_extras(self):
        try:
            import _vade_fontes as F
            extras = F.carregar_normas_extras()
            if extras:
                nomes = ", ".join(
                    (m.get("norma_nome_popular") or m.get("norma_titulo") or nid)
                    for nid, m in sorted(extras.items()))
                self.lbl_extras.configure(text=f"Normas extras monitoradas: {nomes}")
            else:
                self.lbl_extras.configure(
                    text="Normas extras monitoradas: nenhuma — cole um link do Planalto "
                         "acima para monitorar uma nova norma.")
        except Exception:
            self.lbl_extras.configure(text="Normas extras: (erro ao ler normas_extras.json)")

    def _on_adicionar_norma(self):
        url = (self.var_url_nova.get() or "").strip()
        if "planalto.gov.br" not in url.lower():
            messagebox.showwarning(
                "Link inválido",
                "Cole o link da norma compilada no Planalto — por exemplo:\n"
                "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/L14133.htm")
            return
        self.btn_add.state(["disabled"])
        self.lbl_status.configure(text="Lendo a página do Planalto...")

        def worker():
            try:
                import _vade_fontes as F
                meta = F.detectar_norma_planalto(
                    url, progress=lambda m: self._fila.put(("log", m)))
                self._fila.put(("norma_detectada", meta))
            except Exception as e:  # noqa: BLE001
                self._fila.put(("norma_erro", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _confirmar_norma(self, meta: dict):
        self.btn_add.state(["!disabled"])
        self.lbl_status.configure(text="Pronto.")
        import _vade_fontes as F
        if meta["norma_id"] in F.PLANALTO_URLS:
            messagebox.showinfo(
                "Norma já monitorada",
                f"{meta['norma_titulo']} já faz parte do conjunto fixo do vademécum.")
            return
        nome_pop = (self.var_nome_pop.get() or "").strip()
        if nome_pop:
            meta["norma_nome_popular"] = nome_pop
        ja_extra = meta["norma_id"] in F.carregar_normas_extras()
        ementa = meta.get("ementa") or "(ementa não localizada)"
        if not messagebox.askyesno(
                "Atualizar norma monitorada" if ja_extra else "Confirmar nova norma",
                f"{meta['norma_titulo']}\n"
                + (f"Nome popular: {nome_pop}\n" if nome_pop else "")
                + f"norma_id: {meta['norma_id']}\n\n"
                f"Ementa: {ementa[:400]}\n\n"
                + ("Esta norma já está nas extras — atualizar o registro?" if ja_extra else
                   "Adicionar ao monitoramento do vademécum? O texto integral será "
                   "ingerido na próxima atualização (escopo Planalto).")):
            return
        F.adicionar_norma_extra(meta)
        self.var_url_nova.set("")
        self.var_nome_pop.set("")
        self._atualizar_rotulo_extras()
        self._log(f"Norma registrada: {meta['norma_titulo']} (norma_id={meta['norma_id']}). "
                  "O texto integral entra na próxima 'Atualizar base no Notion' com "
                  "'Planalto (normas)' marcado.")

    # ----------------------------------------------------------------- ações
    def _escopos(self):
        """Combinação dos checkboxes -> lista de escopos do updater."""
        escopos = []
        if self.var_planalto.get():
            escopos.append("planalto")
        if self.var_camara.get():
            escopos.append("camara")
        if self.var_qordem.get():
            escopos.append("qordem")
        return escopos

    def _on_dry(self):
        self._iniciar(dry_run=True)

    def _on_atualizar(self):
        if not messagebox.askyesno(
                "Atualizar base",
                "As divergências com as fontes oficiais serão aplicadas na base "
                "'vademecum' do Notion (o texto do Planalto/Câmara é autoritativo), "
                "com enriquecimento via OpenAI nos dispositivos alterados.\n\nContinuar?"):
            return
        self._iniciar(dry_run=False)

    def _iniciar(self, dry_run: bool):
        if self._rodando:
            return
        escopos = self._escopos()
        if not escopos:
            messagebox.showwarning("Escopo vazio", "Marque ao menos uma fonte "
                                   "(Planalto, Câmara ou Questões de ordem).")
            return
        self._rodando = True
        self.btn_dry.state(["disabled"])
        self.btn_run.state(["disabled"])
        self.prog.start(60)
        self.lbl_status.configure(text="Verificando as fontes oficiais..." if dry_run
                                  else "Atualizando a base no Notion...")
        self._log("=" * 70)
        reindexar = bool(self.var_reindexar.get()) and not dry_run
        refresh = bool(self.var_refresh.get())
        normas = [n for n in (self.var_normas.get() or "").replace(",", " ").split() if n] or None

        def worker():
            try:
                import vademecum_updater
                resumo_total = {"normas": [], "qordem": None}
                for i, escopo in enumerate(escopos):
                    r = vademecum_updater.executar(
                        dry_run=dry_run, escopo=escopo, normas=normas,
                        reindexar=False, refresh_dump=refresh and i == 0,
                        progress=lambda m: self._fila.put(("log", m)))
                    resumo_total["normas"].extend(r.get("normas", []))
                    if r.get("qordem"):
                        resumo_total["qordem"] = r["qordem"]
                if reindexar:
                    from conle_gerador import notion_rag
                    self._fila.put(("log", "\nReindexando RAG (base vademecum)..."))
                    notion_rag.indexar(["vademecum"],
                                       progress=lambda m: self._fila.put(("log", m)))
                self._fila.put(("fim", (dry_run, resumo_total)))
            except Exception:
                self._fila.put(("erro", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------- fim
    def _fim(self, dado):
        dry_run, resumo = dado
        self._parar()
        normas = resumo.get("normas") or []
        subst = sum(r.get("substantivos", 0) for r in normas)
        novos = sum(r.get("novos", 0) for r in normas)
        criadas = sum(r.get("criadas", 0) for r in normas)
        atualizadas = sum(r.get("atualizadas", 0) for r in normas)
        qo = resumo.get("qordem") or {}
        if dry_run:
            self.lbl_status.configure(
                text=f"Dry-run: {subst} divergência(s), {novos} dispositivo(s) novo(s), "
                     f"{qo.get('novos', 0)} QO nova(s).")
            messagebox.showinfo(
                "Verificação concluída",
                f"Dispositivos divergentes da fonte oficial: {subst}\n"
                f"Dispositivos novos nas fontes: {novos}\n"
                f"Questões de ordem novas: {qo.get('novos', 0)}\n\n"
                f"Relatórios sync_*.csv em vademecum_work\\")
        else:
            self.lbl_status.configure(
                text=f"Concluído: {atualizadas} atualizada(s), "
                     f"{criadas + qo.get('criadas', 0)} criada(s).")
            messagebox.showinfo(
                "Atualização concluída",
                f"Dispositivos atualizados: {atualizadas}\n"
                f"Dispositivos criados: {criadas}\n"
                f"Questões de ordem criadas: {qo.get('criadas', 0)}")

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
    VademecumGUI().mainloop()
