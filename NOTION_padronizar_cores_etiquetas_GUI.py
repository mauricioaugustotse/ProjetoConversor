#!/usr/bin/env python3
"""GUI para executar a padronizacao de etiquetas no Notion por fases/colunas.

Uso:
- Execute este arquivo.
- Selecione as colunas desejadas.
- Escolha DRY-RUN ou APPLY.
- Clique em Executar.
"""

from __future__ import annotations

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


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notion - Padronizar Etiquetas")
        self.root.geometry("980x690")

        self.proc: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        self.apply_var = tk.BooleanVar(value=False)
        self.debug_var = tk.BooleanVar(value=False)
        self.prune_var = tk.BooleanVar(value=True)
        self.phase_var = tk.StringVar(value="1")
        self.manual_plan_var = tk.StringVar(value="")
        self.resume_enabled_var = tk.BooleanVar(value=False)
        self.resume_input_var = tk.StringVar(value="")
        self.min_interval_var = tk.StringVar(value="0.20")
        self.page_size_var = tk.StringVar(value="100")
        self.timeout_var = tk.StringVar(value="30")
        self.retries_var = tk.StringVar(value="4")

        self.column_vars: dict[str, tk.BooleanVar] = {
            col: tk.BooleanVar(value=(col in PHASE_1)) for col in ALL_COLUMNS
        }

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=10)

        note = (
            "Fluxo recomendado: Fase 1 (siglaUF, relator, siglaClasse, descricaoClasse) "
            "e depois Fase 2 (nomeMunicipio, partes, advogados)."
        )
        tk.Label(top, text=note, anchor="w", justify="left").pack(fill="x")

        warn = (
            "Observacao: colunas muito grandes podem exigir execucao separada por coluna "
            "por limites da API do Notion."
        )
        tk.Label(top, text=warn, fg="#7a4d00", anchor="w", justify="left").pack(fill="x", pady=(4, 0))

        cols_frame = tk.LabelFrame(self.root, text="Colunas")
        cols_frame.pack(fill="x", padx=12, pady=(4, 8))

        for idx, col in enumerate(ALL_COLUMNS):
            r = idx // 4
            c = idx % 4
            chk = tk.Checkbutton(cols_frame, text=col, variable=self.column_vars[col], anchor="w")
            chk.grid(row=r, column=c, sticky="w", padx=10, pady=6)

        presets = tk.Frame(cols_frame)
        presets.grid(row=3, column=0, columnspan=4, sticky="w", padx=10, pady=(4, 8))
        tk.Button(presets, text="Marcar Fase 1", command=self.select_phase_1).pack(side="left", padx=(0, 6))
        tk.Button(presets, text="Marcar Fase 2", command=self.select_phase_2).pack(side="left", padx=6)
        tk.Button(presets, text="Marcar Todas", command=self.select_all).pack(side="left", padx=6)
        tk.Button(presets, text="Limpar", command=self.clear_all).pack(side="left", padx=6)

        opts = tk.LabelFrame(self.root, text="Opcoes de Execucao")
        opts.pack(fill="x", padx=12, pady=8)

        tk.Checkbutton(opts, text="Apply (gravar no Notion)", variable=self.apply_var).grid(
            row=0, column=0, sticky="w", padx=10, pady=6
        )
        tk.Checkbutton(opts, text="Debug", variable=self.debug_var).grid(
            row=0, column=1, sticky="w", padx=10, pady=6
        )
        tk.Checkbutton(opts, text="Prune etiquetas sem uso", variable=self.prune_var).grid(
            row=0, column=2, sticky="w", padx=10, pady=6
        )

        phase_frame = tk.Frame(opts)
        phase_frame.grid(row=0, column=3, sticky="w", padx=10, pady=6)
        tk.Label(phase_frame, text="Fase:").pack(side="left")
        tk.Radiobutton(phase_frame, text="1", variable=self.phase_var, value="1").pack(side="left")
        tk.Radiobutton(phase_frame, text="2", variable=self.phase_var, value="2").pack(side="left")
        tk.Radiobutton(phase_frame, text="all", variable=self.phase_var, value="all").pack(side="left")

        tk.Label(opts, text="min-interval").grid(row=1, column=0, sticky="w", padx=10)
        tk.Entry(opts, width=10, textvariable=self.min_interval_var).grid(row=1, column=0, sticky="e", padx=10)
        tk.Label(opts, text="page-size").grid(row=1, column=1, sticky="w", padx=10)
        tk.Entry(opts, width=10, textvariable=self.page_size_var).grid(row=1, column=1, sticky="e", padx=10)
        tk.Label(opts, text="timeout").grid(row=1, column=2, sticky="w", padx=10)
        tk.Entry(opts, width=10, textvariable=self.timeout_var).grid(row=1, column=2, sticky="e", padx=10)
        tk.Label(opts, text="retries").grid(row=1, column=3, sticky="w", padx=10)
        tk.Entry(opts, width=10, textvariable=self.retries_var).grid(row=1, column=3, sticky="e", padx=10)

        tk.Label(opts, text="manual-plan-output (CSV)").grid(row=2, column=0, sticky="w", padx=10)
        tk.Entry(opts, textvariable=self.manual_plan_var).grid(row=2, column=1, columnspan=3, sticky="we", padx=10)

        tk.Checkbutton(opts, text="Retomar do checkpoint CSV", variable=self.resume_enabled_var).grid(
            row=3, column=0, sticky="w", padx=10, pady=6
        )
        tk.Entry(opts, textvariable=self.resume_input_var).grid(row=3, column=1, columnspan=2, sticky="we", padx=10)
        tk.Button(opts, text="Selecionar...", command=self.browse_resume_file).grid(
            row=3, column=3, sticky="e", padx=10
        )

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=12, pady=8)
        self.run_btn = tk.Button(actions, text="Executar", command=self.run)
        self.run_btn.pack(side="left")
        tk.Button(actions, text="Executar Fase 1", command=self.run_phase_1).pack(side="left", padx=8)
        tk.Button(actions, text="Executar Fase 2", command=self.run_phase_2).pack(side="left", padx=0)
        tk.Button(actions, text="Retomar Pendencias", command=self.run_resume).pack(side="left", padx=8)
        self.stop_btn = tk.Button(actions, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        self.status_label = tk.Label(actions, text="Pronto.")
        self.status_label.pack(side="left", padx=12)

        self.log = scrolledtext.ScrolledText(self.root, wrap="word", height=22, font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def select_phase_1(self) -> None:
        for col in ALL_COLUMNS:
            self.column_vars[col].set(col in PHASE_1)

    def select_phase_2(self) -> None:
        for col in ALL_COLUMNS:
            self.column_vars[col].set(col in PHASE_2)

    def select_all(self) -> None:
        for var in self.column_vars.values():
            var.set(True)

    def clear_all(self) -> None:
        for var in self.column_vars.values():
            var.set(False)

    def _selected_columns(self) -> list[str]:
        return [col for col in ALL_COLUMNS if self.column_vars[col].get()]

    def _build_command(self) -> list[str]:
        script_path = Path(__file__).resolve().parent / SCRIPT_NAME
        if not script_path.exists():
            raise FileNotFoundError(f"Script nao encontrado: {script_path}")

        selected_cols = self._selected_columns()
        if not selected_cols:
            raise ValueError("Selecione ao menos uma coluna.")

        cmd: list[str] = [sys.executable, "-u", str(script_path), "--only-properties", ",".join(selected_cols)]
        cmd.extend(["--phase", self.phase_var.get().strip() or "all"])
        if self.apply_var.get():
            cmd.append("--apply")
        if not self.prune_var.get():
            cmd.append("--no-prune-unused")
        if self.debug_var.get():
            cmd.append("--debug")
        if self.manual_plan_var.get().strip():
            cmd.extend(["--manual-plan-output", self.manual_plan_var.get().strip()])
        if self.resume_enabled_var.get():
            resume_path = self.resume_input_var.get().strip()
            if not resume_path:
                raise ValueError("Selecione um arquivo de checkpoint CSV para retomada.")
            cmd.extend(["--resume-plan-input", resume_path])

        cmd.extend(["--min-interval", self.min_interval_var.get().strip() or "0.20"])
        cmd.extend(["--page-size", self.page_size_var.get().strip() or "100"])
        cmd.extend(["--timeout", self.timeout_var.get().strip() or "30"])
        cmd.extend(["--retries", self.retries_var.get().strip() or "4"])
        return cmd

    def run_phase_1(self) -> None:
        self.phase_var.set("1")
        self.select_phase_1()
        self.run()

    def run_phase_2(self) -> None:
        self.phase_var.set("2")
        self.select_phase_2()
        self.run()

    def browse_resume_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecionar checkpoint CSV",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.resume_input_var.set(path)
            self.resume_enabled_var.set(True)

    def run_resume(self) -> None:
        self.phase_var.set("all")
        self.select_all()
        self.resume_enabled_var.set(True)
        if not self.resume_input_var.get().strip():
            self.browse_resume_file()
            if not self.resume_input_var.get().strip():
                return
        self.run()

    def run(self) -> None:
        if self.running:
            return
        if self.apply_var.get() and self.phase_var.get() == "2":
            proceed = messagebox.askyesno(
                "Confirmar fase 2",
                (
                    "Na API publica do Notion, colunas com muitas opcoes podem ser ignoradas no APPLY "
                    "(limite de 100 options por chamada).\n\n"
                    "Se isso ocorrer, o script gera um CSV de plano manual.\n\n"
                    "Deseja continuar?"
                ),
            )
            if not proceed:
                return
        try:
            cmd = self._build_command()
        except Exception as exc:
            messagebox.showerror("Erro de configuracao", str(exc))
            return

        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        mode = "APPLY" if self.apply_var.get() else "DRY-RUN"
        self.set_status(f"Executando ({mode})...")
        self.append_log("\n" + "=" * 80 + "\n")
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

    def _on_finished(self, rc: int) -> None:
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if rc == 0:
            self.set_status("Concluido com sucesso.")
            self.append_log("\n[OK] Processo finalizado.\n")
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
