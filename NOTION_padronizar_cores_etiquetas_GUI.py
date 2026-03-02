#!/usr/bin/env python3
"""GUI simplificada para padronizacao de etiquetas no Notion.

Fluxo pensado para uso leigo:
1) Escolher operacao (Fase 1, Fase 2 ou Completo).
2) Selecionar CSV base.
3) Executar.
4) Na Fase 2, ler a fila e acompanhar o resumo.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext


SCRIPT_NAME = "NOTION_padronizar_cores_etiquetas_DJeTSE.py"
PHASE_1 = ("siglaUF", "relator", "siglaClasse", "descricaoClasse")
PHASE_2 = ("nomeMunicipio", "partes", "advogados")
ALL_COLUMNS = PHASE_1 + PHASE_2
QUEUE_REQUIRED_FIELDS = [
    "coluna",
    "bloco",
    "faixa_alfabetica",
    "arquivo_csv",
    "arquivo_payload",
    "arquivo_prompt",
    "status",
    "data_execucao",
    "observacao",
    "total_set_default",
    "total_remove_unused",
]


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in " ".join((value or "").strip().lower().split()) if ch.isalnum())


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notion - Padronizar Etiquetas (Simples)")
        self.root.geometry("1120x820")

        self.proc: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        self.queue_rows: list[dict[str, str]] = []
        self.queue_fieldnames: list[str] = []

        base_dir = Path(__file__).resolve().parent
        source_default = self._detect_default_source_csv(base_dir)
        out_default = base_dir

        self.operation_var = tk.StringVar(value="completo")
        self.apply_var = tk.BooleanVar(value=False)
        self.debug_var = tk.BooleanVar(value=False)
        self.source_csv_var = tk.StringVar(value=str(source_default) if source_default else "")
        self.output_dir_var = tk.StringVar(value=str(out_default))
        self.queue_file_var = tk.StringVar(value=str(base_dir / "phase2_chat_queue.csv"))

        # Avancado (oculto por padrao)
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.min_interval_var = tk.StringVar(value="0.20")
        self.page_size_var = tk.StringVar(value="100")
        self.timeout_var = tk.StringVar(value="30")
        self.retries_var = tk.StringVar(value="4")
        self.manual_plan_var = tk.StringVar(value="")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _detect_default_source_csv(self, base_dir: Path) -> Path | None:
        preferred = base_dir / "DJe - 2ª semana - FEV_26_atualizado.csv"
        if preferred.exists() and preferred.is_file():
            return preferred
        csvs = [p for p in base_dir.glob("*.csv") if p.is_file()]
        if not csvs:
            return None
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csvs[0]

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=10)
        tk.Label(
            top,
            text=(
                "Fluxo recomendado: use 'Completo' para rodar Fase 1 (API) e gerar fila da Fase 2 (Notion Chat). "
                "A Fase 2 gera 1 JSON + 1 prompt por coluna (nomeMunicipio, partes, advogados), sem blocos."
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x")

        # Passo 1
        step1 = tk.LabelFrame(self.root, text="1) Operacao")
        step1.pack(fill="x", padx=12, pady=(2, 6))
        tk.Radiobutton(
            step1,
            text="Fase 1 (API oficial): siglaUF, relator, siglaClasse, descricaoClasse",
            variable=self.operation_var,
            value="fase1",
        ).pack(anchor="w", padx=10, pady=3)
        tk.Radiobutton(
            step1,
            text="Fase 2 (Notion Chat): nomeMunicipio, partes, advogados",
            variable=self.operation_var,
            value="fase2",
        ).pack(anchor="w", padx=10, pady=3)
        tk.Radiobutton(
            step1,
            text="Completo (Fase 1 + Fase 2)",
            variable=self.operation_var,
            value="completo",
        ).pack(anchor="w", padx=10, pady=3)

        flags = tk.Frame(step1)
        flags.pack(fill="x", padx=10, pady=(6, 8))
        tk.Checkbutton(flags, text="Apply (gravar no Notion)", variable=self.apply_var).pack(side="left")
        tk.Checkbutton(flags, text="Debug", variable=self.debug_var).pack(side="left", padx=14)

        # Passo 2
        step2 = tk.LabelFrame(self.root, text="2) CSV base")
        step2.pack(fill="x", padx=12, pady=6)
        tk.Label(step2, text="source-csv").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        tk.Entry(step2, textvariable=self.source_csv_var).grid(row=0, column=1, sticky="we", padx=10, pady=8)
        tk.Button(step2, text="Selecionar...", command=self.browse_source_csv).grid(row=0, column=2, padx=10, pady=8)
        step2.columnconfigure(1, weight=1)

        # Passo 3
        step3 = tk.LabelFrame(self.root, text="3) Saida da Fase 2 (Notion Chat)")
        step3.pack(fill="x", padx=12, pady=6)
        tk.Label(step3, text="Pasta de saida").grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        tk.Entry(step3, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="we", padx=10, pady=(8, 4))
        tk.Button(step3, text="Selecionar...", command=self.browse_output_dir).grid(row=0, column=2, padx=10, pady=(8, 4))

        tk.Label(step3, text="Arquivo da fila").grid(row=1, column=0, sticky="w", padx=10, pady=(4, 8))
        tk.Entry(step3, textvariable=self.queue_file_var).grid(row=1, column=1, sticky="we", padx=10, pady=(4, 8))
        tk.Button(step3, text="Selecionar...", command=self.browse_queue_file).grid(row=1, column=2, padx=10, pady=(4, 8))
        step3.columnconfigure(1, weight=1)

        # Avancado
        adv_wrap = tk.Frame(self.root)
        adv_wrap.pack(fill="x", padx=12, pady=(0, 6))
        tk.Checkbutton(
            adv_wrap,
            text="Mostrar opcoes avancadas",
            variable=self.show_advanced_var,
            command=self._toggle_advanced,
        ).pack(anchor="w")

        self.advanced_frame = tk.LabelFrame(self.root, text="Opcoes avancadas")
        tk.Label(self.advanced_frame, text="min-interval").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.min_interval_var).grid(
            row=0, column=1, sticky="w", padx=10, pady=6
        )
        tk.Label(self.advanced_frame, text="page-size").grid(row=0, column=2, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.page_size_var).grid(
            row=0, column=3, sticky="w", padx=10, pady=6
        )
        tk.Label(self.advanced_frame, text="timeout").grid(row=0, column=4, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.timeout_var).grid(
            row=0, column=5, sticky="w", padx=10, pady=6
        )
        tk.Label(self.advanced_frame, text="retries").grid(row=0, column=6, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.retries_var).grid(
            row=0, column=7, sticky="w", padx=10, pady=6
        )

        tk.Label(self.advanced_frame, text="manual-plan-output").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, textvariable=self.manual_plan_var).grid(
            row=1, column=1, columnspan=6, sticky="we", padx=10, pady=6
        )
        tk.Button(self.advanced_frame, text="Selecionar...", command=self.browse_manual_plan).grid(
            row=1, column=7, padx=10, pady=6
        )

        self.advanced_frame.columnconfigure(1, weight=1)

        # Passo 4
        actions = tk.LabelFrame(self.root, text="4) Executar")
        actions.pack(fill="x", padx=12, pady=6)
        self.run_btn = tk.Button(actions, text="Executar operacao selecionada", command=self.run_selected)
        self.run_btn.pack(side="left", padx=8, pady=8)
        tk.Button(actions, text="Gerar fila Fase 2", command=self.generate_phase2_queue).pack(side="left", padx=4, pady=8)
        tk.Button(actions, text="Ler fila", command=self.load_queue).pack(side="left", padx=4, pady=8)
        tk.Button(actions, text="Abrir pasta de saida", command=self.open_output_folder).pack(side="left", padx=4, pady=8)
        self.stop_btn = tk.Button(actions, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8, pady=8)
        self.status_label = tk.Label(actions, text="Pronto.")
        self.status_label.pack(side="left", padx=12)

        # Fila (somente leitura)
        queue_frame = tk.LabelFrame(self.root, text="Fila Fase 2 (somente leitura)")
        queue_frame.pack(fill="both", expand=False, padx=12, pady=(2, 8))
        queue_top = tk.Frame(queue_frame)
        queue_top.pack(fill="x", padx=8, pady=6)
        self.queue_status_label = tk.Label(queue_top, text="Itens: 0 | Gerados: 0 | Colunas: 0")
        self.queue_status_label.pack(side="left", padx=2)

        self.queue_preview = scrolledtext.ScrolledText(queue_frame, wrap="word", height=7, font=("Consolas", 10))
        self.queue_preview.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.queue_preview.configure(state="disabled")

        # Log
        self.log = scrolledtext.ScrolledText(self.root, wrap="word", height=14, font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _toggle_advanced(self) -> None:
        if self.show_advanced_var.get():
            self.advanced_frame.pack(fill="x", padx=12, pady=(0, 8))
        else:
            self.advanced_frame.pack_forget()

    def append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path

    def browse_source_csv(self) -> None:
        initial = self.source_csv_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(Path(__file__).resolve().parent)
        path = filedialog.askopenfilename(
            title="Selecionar source-csv (base)",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.source_csv_var.set(path)

    def browse_output_dir(self) -> None:
        initial = self.output_dir_var.get().strip() or str(Path(__file__).resolve().parent)
        path = filedialog.askdirectory(title="Selecionar pasta de saida", initialdir=initial)
        if path:
            self.output_dir_var.set(path)
            self.queue_file_var.set(str(Path(path) / "phase2_chat_queue.csv"))

    def browse_queue_file(self) -> None:
        initial = self.queue_file_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(Path(__file__).resolve().parent)
        path = filedialog.askopenfilename(
            title="Selecionar arquivo da fila",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.queue_file_var.set(path)

    def browse_manual_plan(self) -> None:
        initial = self.manual_plan_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(Path(__file__).resolve().parent)
        path = filedialog.askopenfilename(
            title="Selecionar manual-plan-output",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.manual_plan_var.set(path)

    def _columns_for_operation(self, op: str) -> list[str]:
        if op == "fase1":
            return list(PHASE_1)
        if op == "fase2":
            return list(PHASE_2)
        return list(ALL_COLUMNS)

    def _build_command(self, *, op: str, queue_only: bool = False) -> list[str]:
        script_path = Path(__file__).resolve().parent / SCRIPT_NAME
        if not script_path.exists():
            raise RuntimeError(f"Script nao encontrado: {script_path}")

        src_raw = self.source_csv_var.get().strip()
        if not src_raw:
            raise RuntimeError("Selecione o source-csv (base).")
        src_path = self._resolve_path(src_raw)
        if not src_path.exists() or not src_path.is_file():
            raise RuntimeError(f"source-csv nao encontrado: {src_path}")

        cols = self._columns_for_operation(op)
        phase = "all"
        phase2_mode = "notion-chat"
        if op == "fase1":
            phase = "1"
            phase2_mode = "api"
        elif op == "fase2":
            phase = "2"
            phase2_mode = "notion-chat"

        cmd = [sys.executable, "-u", str(script_path), "--only-properties", ",".join(cols), "--phase", phase]
        cmd.extend(["--phase2-mode", phase2_mode, "--source-csv", str(src_path)])

        if self.apply_var.get() and not queue_only:
            cmd.append("--apply")
        if self.debug_var.get():
            cmd.append("--debug")

        if self.manual_plan_var.get().strip():
            cmd.extend(["--manual-plan-output", self.manual_plan_var.get().strip()])

        # Avancado
        cmd.extend(["--min-interval", self.min_interval_var.get().strip() or "0.20"])
        cmd.extend(["--page-size", self.page_size_var.get().strip() or "100"])
        cmd.extend(["--timeout", self.timeout_var.get().strip() or "30"])
        cmd.extend(["--retries", self.retries_var.get().strip() or "4"])

        if phase2_mode == "notion-chat":
            out_dir = self.output_dir_var.get().strip()
            queue_file = self.queue_file_var.get().strip()
            if out_dir:
                cmd.extend(["--phase2-chat-output-dir", out_dir])
            if queue_file:
                cmd.extend(["--phase2-queue-file", queue_file])

        return cmd

    def _launch(self, cmd: list[str], status: str) -> None:
        if self.running:
            return
        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.set_status(status)
        self.append_log("\n" + "=" * 100 + "\n")
        self.append_log("Comando: " + " ".join(cmd) + "\n\n")

        def worker() -> None:
            rc = 1
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.root.after(0, self.append_log, line)
                rc = self.proc.wait()
            except Exception as exc:
                self.root.after(0, self.append_log, f"\n[ERRO] {exc}\n")
            finally:
                self.proc = None
                self.root.after(0, self._on_finished, rc)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def run_selected(self) -> None:
        op = self.operation_var.get().strip() or "completo"
        try:
            cmd = self._build_command(op=op, queue_only=False)
        except Exception as exc:
            messagebox.showerror("Erro de configuracao", str(exc))
            return

        if op in {"fase2", "completo"} and self.apply_var.get():
            messagebox.showinfo(
                "Aviso",
                "Na Fase 2 (Notion Chat), o script gera fila/payload/prompt. "
                "Nao ha update de cor via API oficial.",
            )

        self._launch(cmd, "Executando...")

    def generate_phase2_queue(self) -> None:
        try:
            cmd = self._build_command(op="fase2", queue_only=True)
        except Exception as exc:
            messagebox.showerror("Erro de configuracao", str(exc))
            return
        self._launch(cmd, "Gerando fila Fase 2...")

    def _queue_path(self) -> Path:
        raw = self.queue_file_var.get().strip()
        if not raw:
            raise RuntimeError("Informe o arquivo da fila.")
        return self._resolve_path(raw)

    def _ensure_queue_fieldnames(self, fieldnames: list[str]) -> list[str]:
        out = list(fieldnames)
        for name in QUEUE_REQUIRED_FIELDS:
            if name not in out:
                out.append(name)
        return out

    def load_queue(self, show_popup: bool = True) -> None:
        try:
            queue_path = self._queue_path()
        except Exception as exc:
            if show_popup:
                messagebox.showerror("Fila", str(exc))
            return
        if not queue_path.exists() or not queue_path.is_file():
            if show_popup:
                messagebox.showerror("Fila", f"Arquivo de fila nao encontrado: {queue_path}")
            return

        with queue_path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            self.queue_fieldnames = self._ensure_queue_fieldnames([n for n in (reader.fieldnames or []) if n])
            self.queue_rows = [dict(row) for row in reader]

        preview_lines: list[str] = []
        by_status: dict[str, int] = {"GERADO": 0, "PENDENTE": 0, "CONCLUIDO": 0, "ERRO": 0, "OUTROS": 0}
        by_coluna: dict[str, int] = {}
        for idx, row in enumerate(self.queue_rows):
            status = (row.get("status") or "PENDENTE").strip().upper() or "PENDENTE"
            row["status"] = status
            if status in by_status:
                by_status[status] += 1
            else:
                by_status["OUTROS"] += 1
            coluna = (row.get("coluna") or "").strip() or "(sem coluna)"
            by_coluna[coluna] = by_coluna.get(coluna, 0) + 1

            if idx < 40:
                preview_lines.append(
                    f"{idx + 1:03d} | {status:<9} | {coluna:<14} | bloco {row.get('bloco', ''):<3} "
                    f"| faixa {row.get('faixa_alfabetica', ''):<7} | default={row.get('total_set_default', '0')} "
                    f"| rem={row.get('total_remove_unused', '0')}"
                )

        total = len(self.queue_rows)
        generated = by_status["GERADO"]
        self.queue_status_label.config(text=f"Itens: {total} | Gerados: {generated} | Colunas: {len(by_coluna)}")

        summary_lines = ["Resumo por coluna:"]
        for coluna in sorted(by_coluna):
            summary_lines.append(f"- {coluna}: {by_coluna[coluna]} arquivo(s)")
        if total > 40:
            summary_lines.append(f"... mostrando 40 de {total} linhas.")
        rendered_preview = "\n".join(summary_lines + ["", "Primeiras linhas:"] + preview_lines)
        self.queue_preview.configure(state="normal")
        self.queue_preview.delete("1.0", "end")
        self.queue_preview.insert("1.0", rendered_preview)
        self.queue_preview.configure(state="disabled")

        if show_popup:
            messagebox.showinfo("Fila", f"Fila lida: {queue_path}")

    def _open_path(self, path: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return
        subprocess.Popen(["xdg-open", str(path)])

    def open_output_folder(self) -> None:
        raw = self.output_dir_var.get().strip()
        if not raw:
            messagebox.showerror("Pasta", "Informe a pasta de saida.")
            return
        path = self._resolve_path(raw)
        if not path.exists():
            messagebox.showerror("Pasta", f"Pasta nao encontrada: {path}")
            return
        try:
            self._open_path(path)
        except Exception as exc:
            messagebox.showerror("Pasta", f"Falha ao abrir pasta: {exc}")

    def _on_finished(self, rc: int) -> None:
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if rc == 0:
            self.set_status("Concluido com sucesso.")
            self.append_log("\n[OK] Processo finalizado.\n")
            self.load_queue(show_popup=False)
        else:
            self.set_status(f"Concluido com erro (exit={rc}).")
            self.append_log(f"\n[ERRO] Processo finalizado com codigo {rc}.\n")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.append_log("\n[INFO] Solicitado encerramento do processo.\n")
            self.set_status("Encerrando...")

    def on_close(self) -> None:
        if self.running and self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Encerrar", "Existe execucao em andamento. Deseja encerrar?"):
                return
            self.proc.terminate()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
