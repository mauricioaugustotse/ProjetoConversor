"""
Utilitários de GUI (Tkinter) para seleção de arquivos e parâmetros.

Fluxo de trabalho:
1. Normaliza extensões e valida caminhos recebidos.
2. Lista e deduplica arquivos por extensão (com opção recursiva).
3. Exibe painel para seleção de arquivos/pastas e campos extras.
4. Retorna payload padronizado para consumo pelos scripts chamadores.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _normalize_extensions(extensions: Sequence[str]) -> List[str]:
    out: List[str] = []
    for ext in extensions:
        txt = str(ext or "").strip().lower()
        if not txt:
            continue
        if not txt.startswith("."):
            txt = "." + txt
        out.append(txt)
    return sorted(set(out))


def list_files_in_directory(
    folder: str,
    extensions: Sequence[str],
    *,
    recursive: bool = True,
) -> List[str]:
    exts = _normalize_extensions(extensions)
    base = os.path.abspath(os.path.expanduser(folder or ""))
    if not os.path.isdir(base):
        return []

    found: List[str] = []
    if recursive:
        for root, _, files in os.walk(base):
            for name in sorted(files):
                low = name.lower()
                if exts and not any(low.endswith(ext) for ext in exts):
                    continue
                found.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if not os.path.isfile(full):
                continue
            low = name.lower()
            if exts and not any(low.endswith(ext) for ext in exts):
                continue
            found.append(full)

    return dedupe_files(found)


def dedupe_files(paths: Iterable[str], extensions: Optional[Sequence[str]] = None) -> List[str]:
    exts = _normalize_extensions(extensions or [])
    out: List[str] = []
    seen = set()
    for path in paths:
        if not path:
            continue
        full = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(full):
            continue
        low = full.lower()
        if exts and not any(low.endswith(ext) for ext in exts):
            continue
        key = os.path.normcase(full)
        if key in seen:
            continue
        seen.add(key)
        out.append(full)
    return sorted(out, key=lambda p: p.lower())


def open_file_panel(
    *,
    title: str,
    subtitle: str,
    filetypes: Sequence[Tuple[str, str]],
    extensions: Sequence[str],
    initial_files: Optional[Sequence[str]] = None,
    allow_add_dir: bool = True,
    recursive_dir: bool = True,
    min_files: int = 1,
    output_label: str = "",
    initial_output: str = "",
    extra_bools: Optional[Sequence[Tuple[str, str, bool]]] = None,
    extra_texts: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception:
        return None

    exts = _normalize_extensions(extensions)

    class _Panel:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title(title)
            self.root.geometry("920x620")
            self.root.minsize(760, 500)

            self.paths = dedupe_files(initial_files or [], exts)
            self.confirmed = False

            self.status_var = tk.StringVar(value="Selecione arquivos para iniciar.")
            self.output_var = tk.StringVar(value=str(initial_output or ""))

            self.bool_vars: Dict[str, Any] = {}
            self.text_vars: Dict[str, Any] = {}

            self._build()
            self._refresh()

        def _build(self) -> None:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(2, weight=1)

            header = ttk.Frame(self.root, padding=(12, 10, 12, 0))
            header.grid(row=0, column=0, sticky="ew")
            ttk.Label(header, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
            ttk.Label(header, text=subtitle).pack(anchor="w", pady=(2, 0))

            actions = ttk.Frame(self.root, padding=(12, 8))
            actions.grid(row=1, column=0, sticky="ew")
            ttk.Button(actions, text="Adicionar arquivos", command=self._add_files).pack(side="left")
            if allow_add_dir:
                ttk.Button(actions, text="Adicionar pasta", command=self._add_dir).pack(side="left", padx=(8, 0))
            ttk.Button(actions, text="Remover selecionados", command=self._remove_selected).pack(side="left", padx=(8, 0))
            ttk.Button(actions, text="Limpar lista", command=self._clear).pack(side="left", padx=(8, 0))

            list_frame = ttk.Frame(self.root, padding=(12, 0))
            list_frame.grid(row=2, column=0, sticky="nsew")
            list_frame.columnconfigure(0, weight=1)
            list_frame.rowconfigure(0, weight=1)

            self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, activestyle="dotbox")
            self.listbox.grid(row=0, column=0, sticky="nsew")
            scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
            scroll.grid(row=0, column=1, sticky="ns")
            self.listbox.config(yscrollcommand=scroll.set)

            options = ttk.LabelFrame(self.root, text="Opcoes", padding=10)
            options.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 0))
            options.columnconfigure(1, weight=1)
            row_idx = 0

            if output_label:
                ttk.Label(options, text=output_label).grid(row=row_idx, column=0, sticky="w")
                out_entry = ttk.Entry(options, textvariable=self.output_var)
                out_entry.grid(row=row_idx, column=1, sticky="ew", padx=(8, 8))

                def _pick_output() -> None:
                    picked = filedialog.askdirectory(title="Selecione a pasta de saida")
                    if picked:
                        self.output_var.set(os.path.abspath(os.path.expanduser(picked)))

                ttk.Button(options, text="Selecionar...", command=_pick_output).grid(row=row_idx, column=2, sticky="e")
                row_idx += 1

            for key, label, default in (extra_bools or []):
                var = tk.BooleanVar(value=bool(default))
                self.bool_vars[key] = var
                ttk.Checkbutton(options, text=label, variable=var).grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=(6, 0))
                row_idx += 1

            for key, label, default in (extra_texts or []):
                var = tk.StringVar(value=str(default or ""))
                self.text_vars[key] = var
                ttk.Label(options, text=label).grid(row=row_idx, column=0, sticky="w", pady=(6, 0))
                ttk.Entry(options, textvariable=var).grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))
                row_idx += 1

            footer = ttk.Frame(self.root, padding=(12, 8))
            footer.grid(row=4, column=0, sticky="ew")
            ttk.Label(footer, textvariable=self.status_var).pack(anchor="w")

            buttons = ttk.Frame(self.root, padding=(12, 0, 12, 12))
            buttons.grid(row=5, column=0, sticky="ew")
            ttk.Button(buttons, text="Cancelar", command=self._cancel).pack(side="right")
            ttk.Button(buttons, text="Iniciar", command=self._start).pack(side="right", padx=(0, 8))

        def _refresh(self) -> None:
            self.listbox.delete(0, tk.END)
            for path in self.paths:
                self.listbox.insert(tk.END, path)
            self.status_var.set(f"{len(self.paths)} arquivo(s) selecionado(s).")

        def _add_paths(self, paths: Sequence[str]) -> int:
            merged = dedupe_files(list(self.paths) + list(paths), exts)
            before = len(self.paths)
            self.paths = merged
            self._refresh()
            return len(self.paths) - before

        def _add_files(self) -> None:
            picked = filedialog.askopenfilenames(
                title="Selecione arquivos",
                filetypes=list(filetypes) if filetypes else [("Todos", "*.*")],
            )
            if not picked:
                return
            added = self._add_paths(list(picked))
            if added == 0:
                messagebox.showinfo("Sem novos arquivos", "Nenhum novo arquivo foi adicionado.")

        def _add_dir(self) -> None:
            picked = filedialog.askdirectory(title="Selecione uma pasta")
            if not picked:
                return
            files = list_files_in_directory(picked, exts, recursive=recursive_dir)
            if not files:
                messagebox.showwarning("Pasta vazia", "Nenhum arquivo valido foi encontrado na pasta.")
                return
            added = self._add_paths(files)
            if added == 0:
                messagebox.showinfo("Sem novos arquivos", "Todos os arquivos ja estavam na lista.")

        def _remove_selected(self) -> None:
            idxs = list(self.listbox.curselection())
            if not idxs:
                return
            for idx in reversed(idxs):
                del self.paths[idx]
            self._refresh()

        def _clear(self) -> None:
            self.paths = []
            self._refresh()

        def _start(self) -> None:
            if min_files > 0 and len(self.paths) < min_files:
                messagebox.showwarning("Nenhuma entrada", "Selecione ao menos um arquivo para continuar.")
                return
            self.confirmed = True
            self.root.destroy()

        def _cancel(self) -> None:
            self.confirmed = False
            self.root.destroy()

        def run(self) -> Dict[str, Any]:
            self.root.mainloop()
            return {
                "confirmed": self.confirmed,
                "files": list(self.paths),
                "output": str(self.output_var.get()).strip(),
                "bools": {k: bool(v.get()) for k, v in self.bool_vars.items()},
                "texts": {k: str(v.get()).strip() for k, v in self.text_vars.items()},
            }

    panel = _Panel()
    return panel.run()
