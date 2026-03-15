#!/usr/bin/env python3
"""GUI simples para padronizacao de etiquetas no Notion (FASE 1).

Escopo desta GUI:
- Executa apenas a Fase 1 do script principal:
  siglaUF, relator, siglaClasse, descricaoClasse.

Fase 2 foi separada para o script:
- NOTION_corrigir_etiquetas.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
root_path = str(PROJECT_ROOT)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Artefatos.scripts.project_layout import DATA_CSV_DIR, PHASE2_MANUAL_PLAN_DIR, resolve_project_path

SCRIPT_NAME = "NOTION_padronizar_cores_etiquetas_DJeTSE.py"
PHASE_1_COLUMNS = ("siglaUF", "relator", "siglaClasse", "descricaoClasse")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notion - Padronizar Etiquetas (Fase 1)")
        self.root.geometry("1020x760")

        self.proc: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        source_default = self._detect_default_source_csv(DATA_CSV_DIR)

        self.apply_var = tk.BooleanVar(value=False)
        self.debug_var = tk.BooleanVar(value=False)
        self.source_csv_var = tk.StringVar(value=str(source_default) if source_default else "")

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
                "Esta GUI executa somente a Fase 1 via API oficial: "
                "siglaUF, relator, siglaClasse, descricaoClasse."
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x")
        tk.Label(
            top,
            text=(
                "Fase 2 (nomeMunicipio, partes, advogados): usar NOTION_corrigir_etiquetas.py "
                "para gerar JSON/TXT de Notion Chat."
            ),
            anchor="w",
            justify="left",
            fg="#8a5a00",
        ).pack(fill="x", pady=(2, 0))

        step1 = tk.LabelFrame(self.root, text="1) CSV base")
        step1.pack(fill="x", padx=12, pady=6)
        tk.Label(step1, text="source-csv").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        tk.Entry(step1, textvariable=self.source_csv_var).grid(row=0, column=1, sticky="we", padx=10, pady=8)
        tk.Button(step1, text="Selecionar...", command=self.browse_source_csv).grid(row=0, column=2, padx=10, pady=8)
        step1.columnconfigure(1, weight=1)

        step2 = tk.LabelFrame(self.root, text="2) Execucao")
        step2.pack(fill="x", padx=12, pady=6)
        tk.Checkbutton(step2, text="Apply (gravar no Notion)", variable=self.apply_var).pack(side="left", padx=10, pady=8)
        tk.Checkbutton(step2, text="Debug", variable=self.debug_var).pack(side="left", padx=8, pady=8)

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
        tk.Entry(self.advanced_frame, width=10, textvariable=self.min_interval_var).grid(row=0, column=1, sticky="w", padx=10, pady=6)
        tk.Label(self.advanced_frame, text="page-size").grid(row=0, column=2, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.page_size_var).grid(row=0, column=3, sticky="w", padx=10, pady=6)
        tk.Label(self.advanced_frame, text="timeout").grid(row=0, column=4, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.timeout_var).grid(row=0, column=5, sticky="w", padx=10, pady=6)
        tk.Label(self.advanced_frame, text="retries").grid(row=0, column=6, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.retries_var).grid(row=0, column=7, sticky="w", padx=10, pady=6)

        tk.Label(self.advanced_frame, text="manual-plan-output (opcional)").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, textvariable=self.manual_plan_var).grid(row=1, column=1, columnspan=6, sticky="we", padx=10, pady=6)
        tk.Button(self.advanced_frame, text="Selecionar...", command=self.browse_manual_plan).grid(row=1, column=7, padx=10, pady=6)
        self.advanced_frame.columnconfigure(1, weight=1)

        actions = tk.LabelFrame(self.root, text="3) Executar")
        actions.pack(fill="x", padx=12, pady=6)
        self.run_btn = tk.Button(actions, text="Executar Fase 1", command=self.run_phase1)
        self.run_btn.pack(side="left", padx=8, pady=8)
        tk.Button(actions, text="Abrir pasta do projeto", command=self.open_project_folder).pack(side="left", padx=4, pady=8)
        self.stop_btn = tk.Button(actions, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8, pady=8)
        self.status_label = tk.Label(actions, text="Pronto.")
        self.status_label.pack(side="left", padx=12)

        self.log = scrolledtext.ScrolledText(self.root, wrap="word", height=26, font=("Consolas", 10))
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
        return resolve_project_path(raw)

    def browse_source_csv(self) -> None:
        initial = self.source_csv_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(DATA_CSV_DIR)
        path = filedialog.askopenfilename(
            title="Selecionar source-csv (base)",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.source_csv_var.set(path)

    def browse_manual_plan(self) -> None:
        initial = self.manual_plan_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(PHASE2_MANUAL_PLAN_DIR)
        path = filedialog.asksaveasfilename(
            title="Selecionar manual-plan-output",
            initialdir=initial_dir,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.manual_plan_var.set(path)

    def _build_phase1_command(self) -> list[str]:
        script_path = Path(__file__).resolve().parent / SCRIPT_NAME
        if not script_path.exists():
            raise RuntimeError(f"Script nao encontrado: {script_path}")

        src_raw = self.source_csv_var.get().strip()
        if not src_raw:
            raise RuntimeError("Selecione o source-csv (base).")
        src_path = self._resolve_path(src_raw)
        if not src_path.exists() or not src_path.is_file():
            raise RuntimeError(f"source-csv nao encontrado: {src_path}")

        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--only-properties",
            ",".join(PHASE_1_COLUMNS),
            "--phase",
            "1",
            "--source-csv",
            str(src_path),
            "--min-interval",
            self.min_interval_var.get().strip() or "0.20",
            "--page-size",
            self.page_size_var.get().strip() or "100",
            "--timeout",
            self.timeout_var.get().strip() or "30",
            "--retries",
            self.retries_var.get().strip() or "4",
        ]

        if self.apply_var.get():
            cmd.append("--apply")
        if self.debug_var.get():
            cmd.append("--debug")

        manual_out = self.manual_plan_var.get().strip()
        if manual_out:
            cmd.extend(["--manual-plan-output", manual_out])

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

    def run_phase1(self) -> None:
        try:
            cmd = self._build_phase1_command()
        except Exception as exc:
            messagebox.showerror("Erro de configuracao", str(exc))
            return
        self._launch(cmd, "Executando Fase 1...")

    def open_project_folder(self) -> None:
        path = resolve_project_path("")
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Pasta", f"Falha ao abrir pasta: {exc}")

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
