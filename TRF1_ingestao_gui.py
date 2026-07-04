# -*- coding: utf-8 -*-
"""GUI de ingestão de novos boletins do TRF1.

Fluxo: escolher pasta de PDFs -> detectar quais boletins ainda NÃO estão na base Notion
'trf1' -> botão "Ingerir selecionados" que roda, em sequência (subprocess, com log
streamado): (a) pipeline PDF->CSV (A_TRF1, notícias via Gemini); (b) import CSV->Notion
(_trf1_importar); (c) enriquecimento IA dos novos (_trf1_enriquecer --faltantes);
(d) reindexação da RAG (notion_rag --indexar --bases trf1). Checkboxes desligam etapas.

Padrão de threading/log herdado de DJE_relatorios_semanais_gui.py. Roda:
  python TRF1_ingestao_gui.py
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
A_TRF1 = PROJECT_ROOT / "conversores" / "A_TRF1_pdf_to_csv_viaAPI_FINAL.py"
IMPORTAR = PROJECT_ROOT / "_trf1_importar.py"
ENRIQUECER = PROJECT_ROOT / "_trf1_enriquecer.py"
PASTA_DEFAULT = PROJECT_ROOT / "Boletins TRF1"
_INFO_RE = re.compile(r"Bij[_-]?(\d+)", re.IGNORECASE)


def extrair_informativo(path: Path) -> str:
    m = _INFO_RE.search(path.stem)
    return m.group(1) if m else ""


def listar_pdfs(pasta: Path) -> List[Path]:
    return sorted(pasta.rglob("*.pdf")) if pasta.is_dir() else []


def informativos_na_base() -> set:
    """Números de boletim (informativo) já presentes na base Notion.

    'informativo' é um select: as opções do schema equivalem aos valores
    distintos da base e chegam num único GET (vs. paginar ~7,7k linhas).
    Limitação: opção órfã (páginas removidas) ainda conta como "já na base".
    """
    import _trf1_lib as L
    db = L.req("GET", f"https://api.notion.com/v1/databases/{L.DB_ID}")
    opts = db["properties"]["informativo"]["select"]["options"]
    return {str(o.get("name", "")).strip() for o in opts if str(o.get("name", "")).strip()}


def run_command(cmd: List[str], log: Callable[[str], None], env: Dict[str, str] | None = None) -> int:
    """Roda um subprocess na raiz do projeto, transmitindo stdout linha a linha ao log."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    log(f"[exit {proc.returncode}]")
    return proc.returncode


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("TRF1 — Ingestão de boletins")
    root.geometry("940x680")
    root.minsize(820, 600)
    _ico = PROJECT_ROOT / "icones" / "trf1.ico"
    if _ico.exists():
        try:
            root.iconbitmap(str(_ico))
        except Exception:
            pass

    log_queue: "queue.Queue[str]" = queue.Queue()
    busy = tk.BooleanVar(value=False)
    pasta_var = tk.StringVar(value=str(PASTA_DEFAULT))
    status_var = tk.StringVar(value="Escolha a pasta de PDFs e clique em Detectar.")
    opt_importar = tk.BooleanVar(value=True)
    opt_enriquecer = tk.BooleanVar(value=True)
    opt_reindexar = tk.BooleanVar(value=True)
    action_buttons: List[Any] = []
    # id do item na treeview -> {"path", "info", "novo", "marcado"}
    itens: Dict[str, Dict[str, Any]] = {}

    def log(msg: str) -> None:
        log_queue.put(msg)

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)
    main.rowconfigure(5, weight=1)

    # --- 1) Pasta
    src = ttk.LabelFrame(main, text="1) Pasta de PDFs (boletins Bij_NNN.pdf)")
    src.grid(row=0, column=0, sticky="ew")
    src.columnconfigure(0, weight=1)
    ttk.Entry(src, textvariable=pasta_var).grid(row=0, column=0, sticky="ew", padx=6, pady=6)
    ttk.Button(src, text="Escolher pasta…", command=lambda: _escolher_pasta()).grid(row=0, column=1, padx=4)
    btn_detectar = ttk.Button(src, text="Detectar novos", command=lambda: _detectar())
    btn_detectar.grid(row=0, column=2, padx=(4, 6))
    action_buttons.append(btn_detectar)

    # --- 2) Seleção
    selbar = ttk.Frame(main)
    selbar.grid(row=1, column=0, sticky="ew", pady=(8, 2))
    ttk.Button(selbar, text="Marcar novos", command=lambda: _marcar("novos")).pack(side="left")
    ttk.Button(selbar, text="Marcar todos", command=lambda: _marcar("todos")).pack(side="left", padx=4)
    ttk.Button(selbar, text="Limpar", command=lambda: _marcar("nenhum")).pack(side="left")
    conta_var = tk.StringVar(value="")
    ttk.Label(selbar, textvariable=conta_var).pack(side="right")

    tree_box = ttk.Frame(main)
    tree_box.grid(row=2, column=0, sticky="nsew")
    tree_box.columnconfigure(0, weight=1)
    tree_box.rowconfigure(0, weight=1)
    tree = ttk.Treeview(tree_box, columns=("info", "status"), show="tree headings", height=10)
    tree.heading("#0", text="Arquivo")
    tree.heading("info", text="Boletim")
    tree.heading("status", text="Situação")
    tree.column("#0", width=520)
    tree.column("info", width=90, anchor="center")
    tree.column("status", width=160, anchor="center")
    tree.tag_configure("novo", foreground="#1a7f37")
    tree.tag_configure("existe", foreground="#8a8a8a")
    tree.grid(row=0, column=0, sticky="nsew")
    tsc = ttk.Scrollbar(tree_box, orient="vertical", command=tree.yview)
    tsc.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=tsc.set)

    def _mark_symbol(marcado: bool) -> str:
        return "☑ " if marcado else "☐ "

    def _refresh_row(iid: str) -> None:
        it = itens[iid]
        nome = Path(it["path"]).name
        tree.item(iid, text=_mark_symbol(it["marcado"]) + nome,
                  values=(it["info"], "novo" if it["novo"] else "já na base"),
                  tags=("novo" if it["novo"] else "existe",))

    def _on_click(event: Any) -> None:
        iid = tree.identify_row(event.y)
        if iid and iid in itens:
            itens[iid]["marcado"] = not itens[iid]["marcado"]
            _refresh_row(iid)
            _update_count()

    tree.bind("<Button-1>", _on_click)

    def _update_count() -> None:
        marc = sum(1 for it in itens.values() if it["marcado"])
        novos = sum(1 for it in itens.values() if it["novo"])
        conta_var.set(f"{len(itens)} PDFs | {novos} novos | {marc} marcados")

    def _marcar(modo: str) -> None:
        for iid, it in itens.items():
            it["marcado"] = (modo == "todos") or (modo == "novos" and it["novo"])
            _refresh_row(iid)
        _update_count()

    # --- 3) Ações
    acts = ttk.LabelFrame(main, text="2) Ingestão")
    acts.grid(row=3, column=0, sticky="ew", pady=(8, 2))
    ttk.Checkbutton(acts, text="Importar no Notion", variable=opt_importar).grid(row=0, column=0, padx=6, pady=4, sticky="w")
    ttk.Checkbutton(acts, text="Enriquecer com IA (Resumo/Palavras-chave)", variable=opt_enriquecer).grid(row=0, column=1, padx=6, sticky="w")
    ttk.Checkbutton(acts, text="Reindexar RAG", variable=opt_reindexar).grid(row=0, column=2, padx=6, sticky="w")
    btn_ingerir = ttk.Button(acts, text="▶ Ingerir selecionados", command=lambda: _ingerir())
    btn_ingerir.grid(row=0, column=3, padx=10, pady=4)
    action_buttons.append(btn_ingerir)

    # --- 4) Execução
    run_box = ttk.LabelFrame(main, text="3) Execução")
    run_box.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
    run_box.columnconfigure(0, weight=1)
    run_box.rowconfigure(2, weight=1)
    progress = ttk.Progressbar(run_box, mode="indeterminate")
    progress.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
    ttk.Label(run_box, textvariable=status_var).grid(row=1, column=0, sticky="w", padx=6)
    log_widget = tk.Text(run_box, height=12, wrap="word", state="disabled")
    log_widget.grid(row=2, column=0, sticky="nsew", padx=6, pady=(2, 6))
    lsc = ttk.Scrollbar(run_box, orient="vertical", command=log_widget.yview)
    lsc.grid(row=2, column=1, sticky="ns")
    log_widget.configure(yscrollcommand=lsc.set)

    def set_busy(value: bool, status_text: str = "") -> None:
        busy.set(value)
        state = "disabled" if value else "normal"
        for b in action_buttons:
            b.configure(state=state)
        if value:
            progress.start(12)
        else:
            progress.stop()
            progress["value"] = 0
        if status_text:
            status_var.set(status_text)

    def _run_in_thread(target: Callable[[], None], done_message: str) -> None:
        def worker() -> None:
            try:
                target()
                root.after(0, lambda: set_busy(False, done_message))
            except Exception as exc:  # noqa: BLE001
                log(f"ERRO: {exc}")
                root.after(0, lambda: set_busy(False, "Falhou. Veja o log."))
        threading.Thread(target=worker, daemon=True).start()

    def _escolher_pasta() -> None:
        d = filedialog.askdirectory(title="Pasta com os PDFs dos boletins")
        if d:
            pasta_var.set(d)

    def _detectar() -> None:
        if busy.get():
            return
        pasta = Path(pasta_var.get().strip())
        if not pasta.is_dir():
            messagebox.showwarning("Pasta inválida", "Escolha uma pasta existente.")
            return
        set_busy(True, "Detectando boletins na base…")

        def job() -> None:
            pdfs = listar_pdfs(pasta)
            log(f"{len(pdfs)} PDFs encontrados em {pasta}")
            try:
                na_base = informativos_na_base()
                log(f"{len(na_base)} boletins já na base Notion.")
            except Exception as exc:  # noqa: BLE001
                log(f"AVISO: não consegui consultar a base ({exc}). Tratando todos como novos.")
                na_base = set()

            linhas = []
            for p in pdfs:
                info = extrair_informativo(p)
                novo = bool(info) and info not in na_base
                linhas.append((p, info, novo))
            # Novos primeiro (por número do boletim), depois os que já estão na base.
            linhas.sort(key=lambda t: (not t[2], int(t[1]) if t[1].isdigit() else 10**9, str(t[0])))

            def preencher() -> None:
                tree.delete(*tree.get_children())
                itens.clear()
                for p, info, novo in linhas:
                    iid = tree.insert("", "end")
                    itens[iid] = {"path": str(p), "info": info, "novo": novo, "marcado": novo}
                    _refresh_row(iid)
                _update_count()
            root.after(0, preencher)
        _run_in_thread(job, "Detecção concluída.")

    def _ingerir() -> None:
        if busy.get():
            return
        marcados = [it for it in itens.values() if it["marcado"]]
        if not marcados:
            messagebox.showinfo("Nada marcado", "Marque ao menos um boletim (clique nas linhas).")
            return
        pdfs = [it["path"] for it in marcados]
        infos = sorted({it["info"] for it in marcados if it["info"]})
        if not messagebox.askyesno(
            "Confirmar ingestão",
            f"Processar {len(pdfs)} PDF(s) (boletins {', '.join(infos) or '?'}) e enviar ao Notion?",
        ):
            return
        set_busy(True, "Ingerindo… (veja o log)")

        def job() -> None:
            py = sys.executable
            # (a) pipeline PDF -> CSV (notícias via Gemini)
            log("== [1/4] Processando PDFs (A→B, notícias via Gemini)…")
            rc = run_command([py, str(A_TRF1), "--no-gui", "--input-files", *pdfs,
                              "--news-provider", "gemini", "--pipeline-profile", "balanced"], log)
            if rc != 0:
                log("Pipeline PDF→CSV falhou; abortando.")
                return
            # (b) importar CSV -> Notion
            if opt_importar.get():
                log("== [2/4] Importando no Notion…")
                cmd = [py, str(IMPORTAR), "--apply"]
                if infos:
                    cmd += ["--informativos", *infos]
                if run_command(cmd, log) != 0:
                    log("Import no Notion falhou; abortando.")
                    return
            else:
                log("== [2/4] Import no Notion desativado (checkbox).")
            # (c) enriquecer IA (só faltantes)
            if opt_importar.get() and opt_enriquecer.get():
                log("== [3/4] Enriquecendo os novos com IA…")
                run_command([py, str(ENRIQUECER), "--faltantes"], log)
            else:
                log("== [3/4] Enriquecimento IA desativado.")
            # (d) reindexar RAG
            if opt_importar.get() and opt_reindexar.get():
                log("== [4/4] Reindexando a RAG (base trf1)…")
                run_command([py, "-m", "conle_gerador.notion_rag", "--indexar", "--bases", "trf1"], log)
            else:
                log("== [4/4] Reindexação desativada.")
            log("== Ingestão finalizada.")
        _run_in_thread(job, "Ingestão concluída.")

    def drain_log() -> None:
        drained = False
        while True:
            try:
                message = log_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            log_widget.configure(state="normal")
            log_widget.insert("end", message + "\n")
            log_widget.see("end")
            log_widget.configure(state="disabled")
        if drained:
            root.update_idletasks()
        root.after(150, drain_log)

    def _on_close() -> None:
        if busy.get() and not messagebox.askyesno("Sair?", "Há uma tarefa em execução. Sair mesmo assim?"):
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    drain_log()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
