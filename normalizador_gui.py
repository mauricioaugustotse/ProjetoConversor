# -*- coding: utf-8 -*-
"""GUI do Normalizador de PDFs do Notion.

Fluxo: informar a página-mãe -> carregar subpáginas -> diagnosticar/gerar prévia
(sem tocar o Notion) -> conferir a prévia no disco -> aplicar nas selecionadas
(substituição in-place com backup + renomeação). Cada etapa roda o
normalizador_core.py como subprocess com log streamado.

Padrão de threading/log herdado de TRF1_ingestao_gui.py. Roda:
  python normalizador_gui.py
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CORE = PROJECT_ROOT / "normalizador_core.py"
WORK = PROJECT_ROOT / "_normalizador_work"
PAGINA_MAE_DEFAULT = "357721955c6481219cfed7b7f306d1b8"  # Ofício — Consulta TSE (estatutos)


def _carrega_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_command(cmd: List[str], log: Callable[[str], None], env: Dict[str, str] | None = None) -> int:
    """Roda um subprocess na raiz do projeto, transmitindo stdout linha a linha ao log."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace",
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
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Normalizador de PDFs do Notion")
    root.geometry("980x720")
    root.minsize(860, 620)
    _ico = PROJECT_ROOT / "icones" / "normalizador.ico"
    if _ico.exists():
        try:
            root.iconbitmap(str(_ico))
        except Exception:
            pass

    def _add_tooltip(widget: Any, text: str) -> None:
        tip: Dict[str, Any] = {"win": None}

        def _show(_event: Any) -> None:
            if tip["win"] is not None:
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{widget.winfo_rootx() + 10}+{widget.winfo_rooty() + widget.winfo_height() + 6}")
            tk.Label(
                win,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                wraplength=460,
                font=("Segoe UI", 9),
            ).pack(ipadx=6, ipady=4)
            tip["win"] = win

        def _hide(_event: Any) -> None:
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", _show)
        widget.bind("<Leave>", _hide)

    log_queue: "queue.Queue[str]" = queue.Queue()
    busy = tk.BooleanVar(value=False)
    mae_var = tk.StringVar(value=PAGINA_MAE_DEFAULT)
    status_var = tk.StringVar(value="Informe a página-mãe e clique em Carregar subpáginas.")
    opt_ia = tk.BooleanVar(value=False)
    opt_revisao = tk.BooleanVar(value=True)
    opt_imagens = tk.BooleanVar(value=False)
    opt_recorte = tk.BooleanVar(value=True)
    opt_renomear = tk.BooleanVar(value=True)
    action_buttons: List[Any] = []
    # id do item na treeview -> {"page_id", "id8", "titulo", "marcado", ...}
    itens: Dict[str, Dict[str, Any]] = {}

    def log(msg: str) -> None:
        log_queue.put(msg)

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)
    main.rowconfigure(5, weight=1)

    # --- 1) Página-mãe
    src = ttk.LabelFrame(main, text="1) Página-mãe no Notion (URL ou ID)")
    src.grid(row=0, column=0, sticky="ew")
    src.columnconfigure(0, weight=1)
    ent_mae = ttk.Entry(src, textvariable=mae_var)
    ent_mae.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
    btn_carregar = ttk.Button(src, text="Carregar subpáginas", command=lambda: _carregar())
    btn_carregar.grid(row=0, column=1, padx=(4, 6))
    action_buttons.append(btn_carregar)
    _add_tooltip(ent_mae,
                 "Qualquer página do Notion cujas SUBPÁGINAS serão normalizadas\n"
                 "(páginas importadas de PDF, de qualquer tipo de documento).\n"
                 "Aceita a URL completa ou só o ID (32 hex).")
    _add_tooltip(btn_carregar,
                 "Lista as subpáginas da página-mãe e preenche a tabela abaixo\n"
                 "(com o status salvo de execuções anteriores). Não altera nada no Notion.")

    # --- 2) Seleção
    selbar = ttk.Frame(main)
    selbar.grid(row=1, column=0, sticky="ew", pady=(8, 2))
    btn_m_todas = ttk.Button(selbar, text="Marcar todas", command=lambda: _marcar("todas"))
    btn_m_todas.pack(side="left")
    btn_m_pend = ttk.Button(selbar, text="Marcar pendentes", command=lambda: _marcar("pendentes"))
    btn_m_pend.pack(side="left", padx=4)
    btn_m_limpar = ttk.Button(selbar, text="Limpar", command=lambda: _marcar("nenhuma"))
    btn_m_limpar.pack(side="left")
    _add_tooltip(btn_m_todas, "Marca todas as páginas da lista (inclusive as já aplicadas).")
    _add_tooltip(btn_m_pend, "Marca só as que ainda NÃO foram aplicadas — a escolha do dia a dia.")
    _add_tooltip(btn_m_limpar, "Desmarca todas.")
    conta_var = tk.StringVar(value="")
    ttk.Label(selbar, textvariable=conta_var).pack(side="right")

    tree_box = ttk.Frame(main)
    tree_box.grid(row=2, column=0, sticky="nsew")
    tree_box.columnconfigure(0, weight=1)
    tree_box.rowconfigure(0, weight=1)
    tree = ttk.Treeview(tree_box, columns=("partido", "artigos", "ruido", "status"),
                        show="tree headings", height=12)
    tree.heading("#0", text="Página")
    tree.heading("partido", text="Detecção")
    tree.heading("artigos", text="Artigos")
    tree.heading("ruido", text="Ruído (latex/imgs)")
    tree.heading("status", text="Status")
    tree.column("#0", width=430)
    tree.column("partido", width=110, anchor="center")
    tree.column("artigos", width=70, anchor="center")
    tree.column("ruido", width=120, anchor="center")
    tree.column("status", width=110, anchor="center")
    tree.tag_configure("previa", foreground="#1a7f37")
    tree.tag_configure("aplicada", foreground="#8a8a8a")
    tree.tag_configure("atencao", foreground="#b45309")
    tree.grid(row=0, column=0, sticky="nsew")
    tsc = ttk.Scrollbar(tree_box, orient="vertical", command=tree.yview)
    tsc.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=tsc.set)
    _add_tooltip(
        tree,
        "Clique numa linha para marcar/desmarcar (☑/☐) — as AÇÕES em lote usam as marcadas;\n"
        "'Abrir prévia' e 'Restaurar' usam a linha SELECIONADA (destacada).\n\n"
        "Detecção: identificação automática do documento (hoje: sigla de estatuto partidário;\n"
        "   vazio para outros tipos de documento — nada é renomeado nesse caso).\n"
        "Artigos: dispositivos (Art./Artigo) contados após a limpeza — em documento normativo,\n"
        "   valor baixo demais indica página escaneada; em texto comum, fica 0 mesmo.\n"
        "Ruído: ocorrências de LaTeX de OCR / imagens no original.\n"
        "Status: 'previa' (verde) = pronta para conferir e aplicar; 'aplicada' (cinza) = já normalizada no Notion.")

    def _mark_symbol(marcado: bool) -> str:
        return "☑ " if marcado else "☐ "

    def _refresh_row(iid: str) -> None:
        it = itens[iid]
        status = it.get("status", "")
        tags = ()
        if status == "aplicada":
            tags = ("aplicada",)
        elif status == "previa":
            tags = ("previa",)
        if it.get("conferir_partido") and status != "aplicada":
            tags = ("atencao",)
        ruido = ""
        if it.get("latex") is not None:
            ruido = f"{it.get('latex', '?')} / {it.get('imagens', '?')}"
        tree.item(iid, text=_mark_symbol(it["marcado"]) + it["titulo"],
                  values=(it.get("partido", ""), it.get("artigos", ""), ruido, status),
                  tags=tags)

    def _on_click(event: Any) -> None:
        iid = tree.identify_row(event.y)
        if iid and iid in itens:
            itens[iid]["marcado"] = not itens[iid]["marcado"]
            _refresh_row(iid)
            _update_count()

    tree.bind("<Button-1>", _on_click)

    def _update_count() -> None:
        marc = sum(1 for it in itens.values() if it["marcado"])
        aplicadas = sum(1 for it in itens.values() if it.get("status") == "aplicada")
        conta_var.set(f"{len(itens)} páginas | {aplicadas} aplicadas | {marc} marcadas")

    def _marcar(modo: str) -> None:
        for iid, it in itens.items():
            if modo == "todas":
                it["marcado"] = True
            elif modo == "pendentes":
                it["marcado"] = it.get("status") != "aplicada"
            else:
                it["marcado"] = False
            _refresh_row(iid)
        _update_count()

    def _preencher_arvore() -> None:
        paginas = _carrega_json(WORK / "paginas.json", {}).get("paginas", [])
        manifest = _carrega_json(WORK / "manifest.json", {})
        diag = _carrega_json(WORK / "diagnostico.json", {})
        tree.delete(*tree.get_children())
        itens.clear()
        for p in paginas:
            pid = p["page_id"].replace("-", "")
            pid_fmt = f"{pid[0:8]}-{pid[8:12]}-{pid[12:16]}-{pid[16:20]}-{pid[20:32]}"
            man = manifest.get(pid_fmt, {})
            dg = diag.get(pid_fmt, {})
            iid = tree.insert("", "end")
            itens[iid] = {
                "page_id": p["page_id"], "id8": p["id8"],
                "titulo": man.get("titulo_novo_proposto") or p["titulo"],
                "marcado": man.get("status") != "aplicada",
                "status": man.get("status", ""),
                "partido": man.get("partido") or dg.get("partido", ""),
                "conferir_partido": man.get("conferir_partido", False),
                "artigos": dg.get("artigos", ""),
                "latex": dg.get("latex"), "imagens": dg.get("imagens"),
            }
            _refresh_row(iid)
        _update_count()

    # --- 3) Opções e ações
    acts = ttk.LabelFrame(main, text="2) Opções e ações")
    acts.grid(row=3, column=0, sticky="ew", pady=(8, 2))
    chk_ia = ttk.Checkbutton(acts, text="Curadoria fina com IA", variable=opt_ia)
    chk_ia.grid(row=0, column=0, padx=6, pady=2, sticky="w")
    chk_rev = ttk.Checkbutton(acts, text="Revisão final com IA (texto todo)", variable=opt_revisao)
    chk_rev.grid(row=0, column=1, padx=6, sticky="w")
    chk_img = ttk.Checkbutton(acts, text="Manter imagens", variable=opt_imagens)
    chk_img.grid(row=0, column=2, padx=6, sticky="w")
    chk_rec = ttk.Checkbutton(acts, text="Recortar capa/fecho (material extra)", variable=opt_recorte)
    chk_rec.grid(row=0, column=3, padx=6, sticky="w")
    chk_ren = ttk.Checkbutton(acts, text="Renomear páginas", variable=opt_renomear)
    chk_ren.grid(row=0, column=4, padx=6, sticky="w")
    _add_tooltip(chk_ia,
                 "CURADORIA (pontual): passa o gpt-5.6-luna SÓ nos trechos suspeitos (resíduos\n"
                 "de OCR que as regras não resolveram). O gate proíbe QUALQUER remoção ou\n"
                 "alteração de números — serve para consertar caracteres, não para tirar ruído.")
    _add_tooltip(chk_rev,
                 "REVISÃO FINAL (texto completo): última passada do gpt-5.6-luna em TODOS os\n"
                 "parágrafos — corrige espaçamento e REMOVE ruído incrustado (endereço/rodapé\n"
                 "no meio de um parágrafo, número solto, resto de carimbo/selo).\n"
                 "Gate de segurança: só aceita DELEÇÕES — nenhuma palavra ou número novo pode\n"
                 "entrar; máx. 30% removido por parágrafo; linha só some inteira se for curta.\n"
                 "Custo maior que a curadoria (lê o documento todo), mas é o acabamento final.")
    _add_tooltip(chk_img,
                 "Preserva as imagens da página: baixa antes de limpar e reanexa depois de aplicar.\n"
                 "Desmarcado: as imagens são removidas (adequado quando são só carimbos/assinaturas\n"
                 "do escaneamento). ATENÇÃO: em página majoritariamente escaneada as imagens SÃO o\n"
                 "conteúdo — nesse caso, sem esta opção a aplicação é bloqueada por segurança.")
    _add_tooltip(chk_rec,
                 "Remove o que envolve o documento sem ser parte dele: capa/folha de rosto\n"
                 "(decisões, ementas, ofícios de encaminhamento) no início e certidões,\n"
                 "assinaturas e carimbos no final.\n"
                 "O início do documento é localizado por âncoras (título, TÍTULO I/CAPÍTULO I,\n"
                 "Art. 1º...); se não for identificado com segurança, NADA é cortado e a prévia\n"
                 "avisa. Tudo que sai fica registrado em cortes.md — nada é descartado às cegas.")
    _add_tooltip(chk_ren,
                 "Ao aplicar, renomeia a página com o título detectado no documento.\n"
                 "A detecção automática hoje cobre estatutos partidários ('Estatuto do SIGLA —\n"
                 "aprovado em D.M.AAAA'); para outros documentos, sem detecção segura a página\n"
                 "NÃO é renomeada (linha laranja = detecção incerta, também não renomeia).\n"
                 "O título antigo fica guardado e volta com 'Restaurar página…'.")
    btns = ttk.Frame(acts)
    btns.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 4))
    btn_diag = ttk.Button(btns, text="Diagnosticar", command=lambda: _diagnosticar())
    btn_diag.pack(side="left", padx=(6, 4))
    btn_previa = ttk.Button(btns, text="Gerar prévia", command=lambda: _previa())
    btn_previa.pack(side="left", padx=4)
    btn_abrir = ttk.Button(btns, text="Abrir prévia…", command=lambda: _abrir_previa())
    btn_abrir.pack(side="left", padx=4)
    btn_aplicar = ttk.Button(btns, text="▶ Aplicar nas selecionadas", command=lambda: _aplicar())
    btn_aplicar.pack(side="left", padx=12)
    btn_restaurar = ttk.Button(btns, text="Restaurar página…", command=lambda: _restaurar())
    btn_restaurar.pack(side="right", padx=6)
    action_buttons += [btn_diag, btn_previa, btn_aplicar, btn_restaurar]
    _add_tooltip(btn_diag,
                 "Roda a limpeza EM MEMÓRIA nas páginas marcadas e preenche as colunas\n"
                 "Partido/Artigos/Ruído da tabela. Não grava nada — nem no disco, nem no Notion.\n"
                 "Use para ter um raio-X antes de gerar as prévias.")
    _add_tooltip(btn_previa,
                 "Gera os arquivos de conferência das páginas marcadas em _normalizador_work\\<id>\\:\n"
                 "• bruto.md — o texto como está hoje no Notion;\n"
                 "• limpo.md — como ficará depois da normalização;\n"
                 "• cortes.md — exatamente o que será removido (início/fim);\n"
                 "• relatorio.json — métricas e avisos.\n"
                 "Não toca o Notion. É PRÉ-REQUISITO para Aplicar (o hash da prévia é conferido).")
    _add_tooltip(btn_abrir,
                 "Abre no Explorer a pasta da prévia da linha SELECIONADA\n"
                 "(para ler limpo.md e cortes.md antes de aplicar).")
    _add_tooltip(btn_aplicar,
                 "SUBSTITUI o conteúdo das páginas marcadas no Notion pela versão do limpo.md.\n"
                 "Exige prévia gerada e atual; faz backup automático (backup_original.json),\n"
                 "renomeia o título (se marcado) e pula as já aplicadas.\n"
                 "Reversível página a página com 'Restaurar página…'.")
    _add_tooltip(btn_restaurar,
                 "Devolve a página SELECIONADA ao estado do backup feito na aplicação\n"
                 "(conteúdo e título antigos). Limitação: imagens internas do backup\n"
                 "não são recuperáveis (as URLs do Notion expiram) — use a prévia com\n"
                 "'Manter imagens' se precisar delas.")

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
            root.after(0, _preencher_arvore)
        threading.Thread(target=worker, daemon=True).start()

    def _marcadas() -> List[Dict[str, Any]]:
        return [it for it in itens.values() if it["marcado"]]

    def _selecionada() -> Dict[str, Any] | None:
        sel = tree.selection()
        return itens.get(sel[0]) if sel else None

    def _carregar() -> None:
        if busy.get():
            return
        mae = mae_var.get().strip()
        if not mae:
            messagebox.showwarning("Página-mãe", "Informe a URL ou o ID da página-mãe.")
            return
        set_busy(True, "Carregando subpáginas…")

        def job() -> None:
            run_command([sys.executable, str(CORE), "--descobrir", mae], log)
        _run_in_thread(job, "Subpáginas carregadas.")

    def _diagnosticar() -> None:
        if busy.get():
            return
        alvo = _marcadas()
        if not alvo:
            messagebox.showinfo("Nada marcado", "Marque ao menos uma página (clique nas linhas).")
            return
        set_busy(True, f"Diagnosticando {len(alvo)} página(s)…")

        def job() -> None:
            for it in alvo:
                run_command([sys.executable, str(CORE), "--diagnosticar",
                             "--pagina", it["page_id"]], log)
        _run_in_thread(job, "Diagnóstico concluído.")

    def _previa() -> None:
        if busy.get():
            return
        alvo = _marcadas()
        if not alvo:
            messagebox.showinfo("Nada marcado", "Marque ao menos uma página (clique nas linhas).")
            return
        set_busy(True, f"Gerando prévia de {len(alvo)} página(s)…")

        def job() -> None:
            for it in alvo:
                cmd = [sys.executable, str(CORE), "--previa", "--pagina", it["page_id"]]
                if opt_ia.get():
                    cmd.append("--com-ia")
                if opt_revisao.get():
                    cmd.append("--revisao-ia")
                if opt_imagens.get():
                    cmd.append("--manter-imagens")
                if not opt_recorte.get():
                    cmd.append("--sem-recorte")
                if run_command(cmd, log) != 0:
                    log("Prévia falhou; interrompendo o lote.")
                    return
        _run_in_thread(job, "Prévia(s) gerada(s). Confira limpo.md e cortes.md antes de aplicar.")

    def _abrir_previa() -> None:
        it = _selecionada()
        if it:
            d = WORK / it["id8"]
            os.startfile(str(d if d.is_dir() else WORK))  # noqa: S606
        else:
            os.startfile(str(WORK))  # noqa: S606

    def _aplicar() -> None:
        if busy.get():
            return
        alvo = _marcadas()
        if not alvo:
            messagebox.showinfo("Nada marcado", "Marque ao menos uma página (clique nas linhas).")
            return
        sem_previa = [it for it in alvo if not (WORK / it["id8"] / "limpo.md").exists()]
        if sem_previa:
            messagebox.showwarning(
                "Prévia faltando",
                f"{len(sem_previa)} página(s) marcadas ainda não têm prévia gerada. "
                "Gere a prévia (e confira os cortes) antes de aplicar.")
            return
        if not messagebox.askyesno(
            "Confirmar aplicação",
            f"Substituir o conteúdo de {len(alvo)} página(s) no Notion?\n\n"
            "• backup automático por página (backup_original.json)\n"
            "• os trechos removidos estão em cortes.md de cada prévia\n"
            + ("• as páginas serão renomeadas\n" if opt_renomear.get() else ""),
        ):
            return
        set_busy(True, f"Aplicando em {len(alvo)} página(s)… (veja o log)")

        def job() -> None:
            falhas = []
            for it in alvo:
                cmd = [sys.executable, str(CORE), "--aplicar", "--pagina", it["page_id"]]
                if not opt_renomear.get():
                    cmd.append("--sem-renomear")
                if run_command(cmd, log) != 0:
                    falhas.append(it["titulo"])
            if falhas:
                log(f"== {len(falhas)} página(s) NÃO aplicadas (veja os motivos acima): "
                    + "; ".join(falhas))
            log(f"== Lote concluído: {len(alvo) - len(falhas)}/{len(alvo)} aplicadas.")
        _run_in_thread(job, "Aplicação concluída (veja o log).")

    def _restaurar() -> None:
        if busy.get():
            return
        it = _selecionada()
        if not it:
            messagebox.showinfo("Selecione", "Selecione (clique) a página a restaurar na lista.")
            return
        if not messagebox.askyesno(
            "Restaurar página",
            f"Restaurar '{it['titulo']}' a partir do backup?\n"
            "Imagens internas do backup não são restauráveis (URLs expiradas)."):
            return
        set_busy(True, "Restaurando…")

        def job() -> None:
            run_command([sys.executable, str(CORE), "--restaurar",
                         "--pagina", it["page_id"]], log)
        _run_in_thread(job, "Restauração concluída.")

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
    if (WORK / "paginas.json").exists():
        _preencher_arvore()
    drain_log()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
