import os
import re
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF


# ---------------------------- utilidades (sem alterações) ---------------------------- #

def _normalize_path(p: str) -> str:
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(p))

def _is_bold(span: dict) -> bool:
    return any(k in (span.get("font") or "").lower() for k in ("bold", "black", "heavy"))

def _is_italic(span: dict) -> bool:
    return any(k in (span.get("font") or "").lower() for k in ("italic", "oblique"))

def _spans_to_text(spans: List[dict]) -> str:
    parts: List[str] = []
    for sp in spans:
        t = (sp.get("text") or "").replace("\n", " ")
        if not t:
            continue
        if _is_bold(sp) and _is_italic(sp):
            t = f"***{t}***"
        elif _is_bold(sp):
            t = f"**{t}**"
        elif _is_italic(sp):
            t = f"*{t}*"
        parts.append(t)
    return " ".join(parts)

def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _detect_headers_footers(
    pages_lines: Iterable[Tuple[float, List[Tuple[float, float, str]]]],
    top_ratio: float = 0.12,
    bottom_ratio: float = 0.12,
    thr: float = 0.5,
) -> Tuple[set, set]:
    pages_lines = list(pages_lines)
    n = len(pages_lines)
    if n == 0:
        return set(), set()
    head = Counter()
    foot = Counter()
    for H, lines in pages_lines:
        tcut = H * top_ratio
        bcut = H * (1 - bottom_ratio)
        for y0, y1, t in lines:
            if not t or len(t) > 140:
                continue
            if y0 <= tcut:
                head[t] += 1
            elif y1 >= bcut:
                foot[t] += 1
    headers = {t for t, c in head.items() if c >= thr * n}
    footers = {t for t, c in foot.items() if c >= thr * n}
    return headers, footers

def _unhyphenate(lines: List[str]) -> List[str]:
    out: List[str] = []
    for t in lines:
        if out and out[-1].endswith("-") and (t[:1].islower()):
            out[-1] = out[-1][:-1] + t.lstrip()
        else:
            out.append(t)
    return out

def _process_lines_to_paragraphs(lines: List[str]) -> List[str]:
    paragraphs: List[str] = []
    current: List[str] = []
    bullet_re = re.compile(r"^[\s\u2022\u00b7\u25e6\u2013\u2014\uf0d5\uf076\uf07d\-•·◦–—]+\s*")
    numbered_re = re.compile(r"^\s*((\(?[0-9]+[\)\.]|[ivxlcdmIVXLCDM]+\.)\s+)")
    def flush_current():
        if current:
            paragraphs.append(" ".join(current).strip())
            current.clear()
    for line in lines:
        stripped = line.lstrip()
        if bullet_re.match(stripped):
            flush_current()
            content = bullet_re.sub("", stripped, count=1).strip()
            paragraphs.append(f"- {content}")
        elif numbered_re.match(stripped):
            flush_current()
            m = numbered_re.match(stripped)
            content = stripped[m.end():].strip()
            lead = m.group(1)
            paragraphs.append(f"{lead}{content}")
        else:
            current.append(stripped)
    flush_current()
    return paragraphs

def _slugify(text: str, counts: defaultdict) -> str:
    slug = re.sub(r"[^a-z0-9\- ]+", "", text.lower()).strip().replace(" ", "-")
    counts[slug] += 1
    return slug if counts[slug] == 1 else f"{slug}-{counts[slug]}"


# -------------------------- conversão principal (MODIFICADO) -------------------------- #

def convert_pdf_to_markdown(
    pdf_path: str,
    md_path: str,
) -> None:
    # --- ALTERAÇÃO: Removidos parâmetros 'assets_dir' e 'include_images' ---

    # 0) Caminhos / pastas
    pdf_path = _normalize_path(pdf_path)
    md_path = _normalize_path(md_path)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    # --- ALTERAÇÃO: Removida a criação da pasta de assets ---

    # 1) Primeira passada: tamanhos e linhas
    doc = fitz.open(pdf_path)
    all_sizes: List[float] = []
    pages_lines: List[Tuple[float, List[Tuple[float, float, str]]]] = []
    page_dicts: List[dict] = []

    for page in doc:
        d = page.get_text("dict")
        page_dicts.append(d)
        H = page.rect.height
        lines = []
        for b in d.get("blocks", []):
            for ln in b.get("lines", []):
                spans = ln.get("spans") or []
                if not spans:
                    continue
                y0 = min(s["bbox"][1] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                txt = _ws(_spans_to_text(spans))
                if txt:
                    lines.append((y0, y1, txt))
                for sp in spans:
                    sz = sp.get("size")
                    if isinstance(sz, (int, float)):
                        all_sizes.append(round(float(sz), 1))
        pages_lines.append((H, lines))

    # 2) Cabeçalhos/rodapés repetidos
    headers, footers = _detect_headers_footers(pages_lines)

    # 3) Mapear tamanhos de fonte -> níveis de título
    levels: dict = {}
    for i, sz in enumerate(sorted(set(all_sizes), reverse=True)[:5]):
        levels[sz] = i + 1  # #, ##, ###, ####, #####

    # 4) Construir Markdown
    md_lines: List[str] = []
    anchors: List[Tuple[int, str, str]] = []
    anchor_counts: defaultdict = defaultdict(int)
    md_lines.append(f"# Conteúdo de: {Path(pdf_path).name}\n\n> **TOC** será gerado abaixo.\n\n")

    for pnum, page in enumerate(doc, start=1):
        d = page_dicts[pnum - 1]
        
        # --- ALTERAÇÃO: Bloco inteiro de extração de imagens foi REMOVIDO daqui ---

        try:
            tabs = page.find_tables()
            for tb in getattr(tabs, "tables", []):
                rows = tb.extract() or []
                rows = [[(c or "").replace("|", "\\|") for c in r] for r in rows]
                if rows:
                    md_lines.append("\n| " + " | ".join(rows[0]) + " |\n")
                    md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |\n")
                    for r in rows[1:]:
                        md_lines.append("| " + " | ".join(r) + " |\n")
        except Exception:
            pass
            
        for b in d.get("blocks", []):
            raw_lines: List[str] = []
            sizes_in_block: List[float] = []
            for ln in b.get("lines", []):
                text = _ws(_spans_to_text(ln.get("spans", [])))
                if not text or text in headers or text in footers:
                    continue
                if re.match(r"^\s*---\s*Fim da Página\s*---\s*$", text, re.I):
                    continue
                raw_lines.append(text)
                for sp in ln.get("spans", []):
                    sz = sp.get("size")
                    if isinstance(sz, (int, float)):
                        sizes_in_block.append(round(float(sz), 1))
            if not raw_lines:
                continue
            raw_lines = _unhyphenate(raw_lines)
            paragraphs = _process_lines_to_paragraphs(raw_lines)
            lvl = None
            if sizes_in_block:
                dom = Counter(sizes_in_block).most_common(1)[0][0]
                lvl = levels.get(dom)
            block_text = "\n".join(paragraphs)
            if lvl and len(block_text) < 180:
                slug = _slugify(block_text, anchor_counts)
                anchors.append((lvl, block_text, slug))
                md_lines.append(f"\n{'#' * lvl} {block_text}\n")
            else:
                for para in paragraphs:
                    md_lines.append(para + "\n\n")

    # 5) TOC
    toc_lines: List[str] = []
    for lvl, text, anchor in anchors:
        toc_lines.append(f"{'  ' * (lvl - 1)}- [{text}](#{anchor})")
    if not toc_lines:
        toc_lines.append("- (sem títulos detectados)")
    md_lines.insert(2, "\n".join(toc_lines) + "\n\n")

    # 6) Gravar saída
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    print(f"  [OK] Markdown salvo em: {md_path}")


# ------------------------------ CLI e Lógica de Lote (MODIFICADO) ------------------------------ #

def main():
    """
    Função principal que busca e processa todos os PDFs em uma pasta.
    """
    parser = argparse.ArgumentParser(
        description="Converte todos os arquivos PDF de uma pasta para Markdown, focando apenas no texto.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        default=".",
        help="Pasta contendo os arquivos PDF. Os arquivos .md serão salvos na mesma pasta.\nPadrão: pasta atual."
    )
    # --- ALTERAÇÃO: O argumento "--include-images" foi completamente removido ---
    args = parser.parse_args()

    input_dir = Path(_normalize_path(args.input))

    if not input_dir.is_dir():
        print(f"Erro: A pasta de entrada '{input_dir}' não existe.")
        sys.exit(1)
    
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))

    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em '{input_dir}'.")
        return

    print(f"Encontrados {len(pdf_files)} arquivos PDF. Iniciando conversão de texto...")
    print("-" * 30)

    for pdf_path in pdf_files:
        print(f"Processando: {pdf_path.name}")
        
        md_path = pdf_path.with_suffix(".md")
        
        # --- ALTERAÇÃO: 'assets_dir' não é mais definido ou necessário ---

        try:
            # --- ALTERAÇÃO: Chamada de função simplificada, sem parâmetros de imagem ---
            convert_pdf_to_markdown(
                pdf_path=str(pdf_path),
                md_path=str(md_path)
            )
        except Exception as e:
            print(f"  [ERRO] Falha ao converter '{pdf_path.name}': {e}")
        print("-" * 10)

    print(f"\nConversão concluída! {len(pdf_files)} arquivos processados.")
    print(f"Arquivos salvos na pasta de origem: '{input_dir.resolve()}'")


if __name__ == "__main__":
    main()