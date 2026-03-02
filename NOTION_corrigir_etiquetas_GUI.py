#!/usr/bin/env python3
"""GUI simples para gerar JSON/TXT da Fase 2 (Notion Chat).

Backend:
- NOTION_corrigir_etiquetas.py (preferencial), ou
- NOTION_corrigir_etiquetas_JSON.py (fallback).
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


BACKEND_CANDIDATES = (
    "NOTION_corrigir_etiquetas.py",
    "NOTION_corrigir_etiquetas_JSON.py",
)
DEFAULT_DATABASE_URL = "https://www.notion.so/316721955c6480b4af2cf19fa557a5dd?v=316721955c64816e8f6f000c06433647"
PHASE2_COLUMNS = ("nomeMunicipio", "partes", "advogados")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notion - Corrigir Etiquetas (Fase 2)")
        self.root.geometry("1080x820")

        self.proc: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        base_dir = Path(__file__).resolve().parent

        self.manual_plan_var = tk.StringVar(value=str(base_dir / "notion_etiquetas_plano_manual_fase2.csv"))
        self.output_dir_var = tk.StringVar(value=str(base_dir))
        self.database_url_var = tk.StringVar(value=DEFAULT_DATABASE_URL)

        self.col_nome_var = tk.BooleanVar(value=True)
        self.col_partes_var = tk.BooleanVar(value=True)
        self.col_adv_var = tk.BooleanVar(value=True)

        self.color_mode_var = tk.StringVar(value="default")
        self.include_removals_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar(value=False)

        self.timeout_var = tk.StringVar(value="30")
        self.retries_var = tk.StringVar(value="4")
        self.show_advanced_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        intro = tk.Frame(self.root)
        intro.pack(fill="x", padx=12, pady=10)
        tk.Label(
            intro,
            text=(
                "Esta GUI gera os arquivos da Fase 2 para Notion Chat: "
                "phase2_<coluna>.json, phase2_<coluna>.prompt.txt e phase2_chat_queue.csv"
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x")

        block1 = tk.LabelFrame(self.root, text="1) Entrada")
        block1.pack(fill="x", padx=12, pady=6)

        tk.Label(block1, text="manual-plan-csv").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        tk.Entry(block1, textvariable=self.manual_plan_var).grid(row=0, column=1, sticky="we", padx=10, pady=8)
        tk.Button(block1, text="Selecionar...", command=self.browse_manual_plan).grid(row=0, column=2, padx=10, pady=8)

        tk.Label(block1, text="database-url").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        tk.Entry(block1, textvariable=self.database_url_var).grid(row=1, column=1, sticky="we", padx=10, pady=8)

        block1.columnconfigure(1, weight=1)

        block2 = tk.LabelFrame(self.root, text="2) Colunas da Fase 2")
        block2.pack(fill="x", padx=12, pady=6)

        tk.Checkbutton(block2, text="nomeMunicipio", variable=self.col_nome_var).pack(side="left", padx=12, pady=8)
        tk.Checkbutton(block2, text="partes", variable=self.col_partes_var).pack(side="left", padx=12, pady=8)
        tk.Checkbutton(block2, text="advogados", variable=self.col_adv_var).pack(side="left", padx=12, pady=8)

        block3 = tk.LabelFrame(self.root, text="3) Saida")
        block3.pack(fill="x", padx=12, pady=6)

        tk.Label(block3, text="Pasta de saida").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        tk.Entry(block3, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="we", padx=10, pady=8)
        tk.Button(block3, text="Selecionar...", command=self.browse_output_dir).grid(row=0, column=2, padx=10, pady=8)

        row_flags = tk.Frame(block3)
        row_flags.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))

        tk.Label(row_flags, text="color-mode:").pack(side="left")
        tk.Radiobutton(row_flags, text="default", value="default", variable=self.color_mode_var).pack(side="left", padx=(4, 12))
        tk.Radiobutton(row_flags, text="target", value="target", variable=self.color_mode_var).pack(side="left", padx=(0, 16))

        tk.Checkbutton(row_flags, text="Incluir removals", variable=self.include_removals_var).pack(side="left", padx=(0, 14))
        tk.Checkbutton(row_flags, text="Debug", variable=self.debug_var).pack(side="left")

        block3.columnconfigure(1, weight=1)

        adv_wrap = tk.Frame(self.root)
        adv_wrap.pack(fill="x", padx=12, pady=(0, 6))
        tk.Checkbutton(
            adv_wrap,
            text="Mostrar opcoes avancadas",
            variable=self.show_advanced_var,
            command=self._toggle_advanced,
        ).pack(anchor="w")

        self.advanced_frame = tk.LabelFrame(self.root, text="Opcoes avancadas")
        tk.Label(self.advanced_frame, text="timeout").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.timeout_var).grid(row=0, column=1, sticky="w", padx=10, pady=6)
        tk.Label(self.advanced_frame, text="retries").grid(row=0, column=2, sticky="w", padx=10, pady=6)
        tk.Entry(self.advanced_frame, width=10, textvariable=self.retries_var).grid(row=0, column=3, sticky="w", padx=10, pady=6)

        actions = tk.LabelFrame(self.root, text="4) Executar")
        actions.pack(fill="x", padx=12, pady=6)

        self.run_btn = tk.Button(actions, text="Gerar JSON/TXT Fase 2", command=self.run)
        self.run_btn.pack(side="left", padx=8, pady=8)
        tk.Button(actions, text="Ler fila gerada", command=self.load_queue).pack(side="left", padx=4, pady=8)
        tk.Button(actions, text="Abrir pasta de saida", command=self.open_output_folder).pack(side="left", padx=4, pady=8)

        self.stop_btn = tk.Button(actions, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8, pady=8)

        self.status_label = tk.Label(actions, text="Pronto.")
        self.status_label.pack(side="left", padx=12)

        queue_frame = tk.LabelFrame(self.root, text="Resumo da fila")
        queue_frame.pack(fill="both", expand=False, padx=12, pady=(2, 8))
        self.queue_preview = scrolledtext.ScrolledText(queue_frame, wrap="word", height=7, font=("Consolas", 10))
        self.queue_preview.pack(fill="both", expand=True, padx=8, pady=8)
        self.queue_preview.configure(state="disabled")

        self.log = scrolledtext.ScrolledText(self.root, wrap="word", height=16, font=("Consolas", 10))
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
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        return p

    def _backend_script(self) -> Path:
        base = Path(__file__).resolve().parent
        for name in BACKEND_CANDIDATES:
            candidate = base / name
            if candidate.exists() and candidate.is_file():
                return candidate
        raise RuntimeError(
            "Backend nao encontrado. Esperado um destes arquivos: " + ", ".join(BACKEND_CANDIDATES)
        )

    def _selected_properties(self) -> list[str]:
        props: list[str] = []
        if self.col_nome_var.get():
            props.append("nomeMunicipio")
        if self.col_partes_var.get():
            props.append("partes")
        if self.col_adv_var.get():
            props.append("advogados")
        return props

    def _build_command(self) -> tuple[list[str], Path]:
        script_path = self._backend_script()

        manual_raw = self.manual_plan_var.get().strip()
        if not manual_raw:
            raise RuntimeError("Selecione o arquivo manual-plan-csv.")
        manual_path = self._resolve_path(manual_raw)
        if not manual_path.exists() or not manual_path.is_file():
            raise RuntimeError(f"manual-plan-csv nao encontrado: {manual_path}")

        db_url = self.database_url_var.get().strip()
        if not db_url:
            raise RuntimeError("Informe o database-url.")

        out_raw = self.output_dir_var.get().strip()
        if not out_raw:
            raise RuntimeError("Informe a pasta de saida.")
        out_dir = self._resolve_path(out_raw)
        out_dir.mkdir(parents=True, exist_ok=True)

        props = self._selected_properties()
        if not props:
            raise RuntimeError("Selecione pelo menos uma coluna da fase 2.")

        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--manual-plan-csv",
            str(manual_path),
            "--database-url",
            db_url,
            "--properties",
            ",".join(props),
            "--output-dir",
            str(out_dir),
            "--color-mode",
            self.color_mode_var.get().strip() or "default",
            "--timeout",
            self.timeout_var.get().strip() or "30",
            "--retries",
            self.retries_var.get().strip() or "4",
        ]

        if not self.include_removals_var.get():
            cmd.append("--no-removals")
        if self.debug_var.get():
            cmd.append("--debug")

        return cmd, out_dir

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

    def browse_manual_plan(self) -> None:
        initial = self.manual_plan_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(Path(__file__).resolve().parent)
        path = filedialog.askopenfilename(
            title="Selecionar manual-plan-csv",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.manual_plan_var.set(path)

    def browse_output_dir(self) -> None:
        initial = self.output_dir_var.get().strip() or str(Path(__file__).resolve().parent)
        path = filedialog.askdirectory(title="Selecionar pasta de saida", initialdir=initial)
        if path:
            self.output_dir_var.set(path)

    def run(self) -> None:
        try:
            cmd, _ = self._build_command()
        except Exception as exc:
            messagebox.showerror("Erro de configuracao", str(exc))
            return
        self._launch(cmd, "Gerando JSON/TXT da Fase 2...")

    def _queue_path(self) -> Path:
        out_dir = self._resolve_path(self.output_dir_var.get().strip())
        return out_dir / "phase2_chat_queue.csv"

    def load_queue(self) -> None:
        queue_path = self._queue_path()
        if not queue_path.exists() or not queue_path.is_file():
            messagebox.showerror("Fila", f"Arquivo nao encontrado: {queue_path}")
            return

        with queue_path.open("r", encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp))

        lines = [f"Fila: {queue_path}", f"Itens: {len(rows)}", ""]
        by_col = {}
        for row in rows:
            col = (row.get("coluna") or "(sem coluna)").strip()
            by_col[col] = by_col.get(col, 0) + 1

        lines.append("Resumo por coluna:")
        for col in sorted(by_col):
            lines.append(f"- {col}: {by_col[col]} arquivo(s)")

        lines.append("")
        lines.append("Primeiras linhas:")
        for i, row in enumerate(rows[:30], start=1):
            lines.append(
                f"{i:03d} | {row.get('coluna','')} | options={row.get('total_options','0')} | "
                f"remove={row.get('total_remove_unused','0')} | {row.get('status','')}"
            )

        self.queue_preview.configure(state="normal")
        self.queue_preview.delete("1.0", "end")
        self.queue_preview.insert("1.0", "\n".join(lines))
        self.queue_preview.configure(state="disabled")

        messagebox.showinfo("Fila", f"Fila lida: {queue_path}")

    def open_output_folder(self) -> None:
        try:
            path = self._resolve_path(self.output_dir_var.get().strip())
            if not path.exists():
                raise RuntimeError(f"Pasta nao encontrada: {path}")

            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Pasta", str(exc))

    def _on_finished(self, rc: int) -> None:
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if rc == 0:
            self.set_status("Concluido com sucesso.")
            self.append_log("\n[OK] Processo finalizado.\n")
            try:
                self.load_queue()
            except Exception:
                pass
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
