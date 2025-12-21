# -*- coding: utf-8 -*-
"""
Script AUTOMÁTICO e ESTRUTURADO para converter um arquivo Markdown de legislação
em um arquivo CSV formatado para importação no Notion, respeitando a hierarquia
de artigos, parágrafos, incisos e alíneas.
"""

import csv
import re
import os
import glob


def normalize_ordinals(text):
    """Padroniza indicadores ordinais duplicados e variações inconsistentes."""
    if not text:
        return text
    text = text.replace('.\u00ba', '\u00ba').replace('.\u00aa', '\u00aa')
    text = re.sub(r'\u00ba{2,}', '\u00ba', text)
    text = re.sub(r'\u00aa{2,}', '\u00aa', text)
    text = re.sub(r'(\d+)\s*[oO\u00ba]\b', lambda m: f"{m.group(1)}\u00ba", text)
    return text


def normalize_decimal_commas(text):
    """Remove espaços indevidos em torno de vírgulas decimais."""
    if not text:
        return text
    return re.sub(r'(?<=\d)\s*,\s*(?=\d)', ',', text)


def normalize_space_characters(text):
    """Normaliza espaços especiais (NBSP, ZWSP, BOM) para evitar falhas de reconhecimento."""
    if not text:
        return text
    text = text.replace('\ufeff', '')
    text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\u2060]', '', text)
    return text


article_sequence_pattern = re.compile(
    r'(\barts?\.)'
    r'(\s*)'
    r'(\d[\wºª-]*(?:,\s*\d[\wºª-]*)+)',
    flags=re.IGNORECASE,
)


def fix_article_number_spacing(text):
    """Garante espaço após vírgulas em sequências de artigos (ex.: 'arts. 21, 26')."""
    if not text:
        return text

    def repl(match):
        prefix, spacing, numbers = match.groups()
        spacing = spacing if spacing else ' '
        normalized_numbers = re.sub(r',\s*(?=\d)', ', ', numbers)
        return f"{prefix}{spacing}{normalized_numbers}"

    return article_sequence_pattern.sub(repl, text)


def clean_title(raw_title):
    """Funcao centralizada para limpar os titulos dos dispositivos."""
    cleaned = normalize_space_characters(raw_title)
    cleaned = cleaned.strip()
    cleaned = re.sub(r'(?i)^Art\.\s*', 'Art. ', cleaned)
    cleaned = normalize_ordinals(cleaned)
    cleaned = cleaned.rstrip('.')
    cleaned = re.sub(r'\u00a7\s+', '\u00a7', cleaned)
    cleaned = re.sub(
        r'(\u00a7)([1-9])(?!\u00ba|\d)',
        lambda m: f"{m.group(1)}{m.group(2)}\u00ba",
        cleaned,
    )
    cleaned = re.sub(
        r'(?i)(Lei)\s+n[\s.\-]*(?:\u00ba|o)?\s*(?=\d)',
        lambda m: f"{m.group(1)} n\u00ba ",
        cleaned,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def collapse_duplicate_text(text):
    """Remove duplicações simples em que o mesmo conteúdo aparece duas vezes em sequência."""
    if not text:
        return text

    leading_len = len(text) - len(text.lstrip())
    trailing_len = len(text) - len(text.rstrip())
    core = text[leading_len:len(text) - trailing_len if trailing_len else None]

    starts_with_newline = core.startswith('\n') if core else False
    core_stripped = core.strip()

    if core_stripped:
        duplicate_match = re.match(r'(?P<block>.+?)\s+(?P=block)$', core_stripped, re.DOTALL)
        if duplicate_match:
            core_stripped = duplicate_match.group('block')

    if starts_with_newline:
        core_stripped = '\n' + core_stripped

    prefix = text[:leading_len] if leading_len else ''
    suffix = text[len(text) - trailing_len:] if trailing_len else ''
    return f"{prefix}{core_stripped}{suffix}"



def save_current_item(rows, title, text_lines):
    """Funcao auxiliar para salvar o item (artigo, paragrafo, etc.) processado.
    Agrupa as linhas de texto, remove anotacoes/links/lixo e adiciona a lista de resultados."""
    cleaned_text = None
    title_normalized = clean_title(title) if title else title
    if title and text_lines:
        link_pattern = re.compile(r'\[([^\]]+)\]\s*\([^)]*\)')
        asterisk_header_pattern = re.compile(r'(\s*\*\*[^\*]+\*\*\s*)+')
        bold_inline_pattern = re.compile(r'\*\*(.+?)\*\*')

        normalized_lines = []

        for segment in text_lines:
            if segment is None:
                continue
            segment = normalize_space_characters(segment)
            segment = segment.replace('\r', '')

            for raw_line in segment.split('\n'):
                raw_line = normalize_space_characters(raw_line)
                candidate = raw_line.strip()
                if not candidate:
                    continue
                candidate = re.sub(r'^#{1,6}\s*', '', candidate)

                candidate = link_pattern.sub(r'\1', candidate)
                candidate = re.sub(r'\[([^\]]+)\]', r'\1', candidate)
                candidate = bold_inline_pattern.sub(lambda m: m.group(1).strip(), candidate)
                candidate = asterisk_header_pattern.sub(' ', candidate)

                candidate = candidate.replace('_', '')
                candidate = normalize_ordinals(candidate)
                candidate = re.sub(r'(?i)\bArt\.\s*(?=\d)', 'Art. ', candidate)
                candidate = re.sub(r'\u00a7\s+', '\u00a7', candidate)
                candidate = re.sub(
                    r'(\u00a7)([1-9])(?!\u00ba|\d)',
                    lambda m: f"{m.group(1)}{m.group(2)}\u00ba",
                    candidate,
                )
                candidate = re.sub(
                    r'(?i)(Lei)\s+n[\s.\-]*(?:\u00ba|o)?\s*(?=\d)',
                    lambda m: f"{m.group(1)} n\u00ba ",
                    candidate,
                )
                candidate = re.sub(r'\(\s*(?:https?://|\.\.?/|#)[^)]+\)', '', candidate)
                candidate = re.sub(r'\(\s*[\w\u00C0-\u024F./#%-]*\.html?[^)]*\)', '', candidate, flags=re.IGNORECASE)
                candidate = re.sub(r'\s*\(\s*', ' (', candidate)
                candidate = re.sub(r'\s+\)', ')', candidate)
                candidate = re.sub(r'\)\s*(?=\()', ') (', candidate)
                candidate = re.sub(r'^ \(', '(', candidate)
                candidate = re.sub(r'\(\s*\(', '(', candidate)
                candidate = re.sub(r'\)\s*\)', ')', candidate)
                candidate = re.sub(r'\[\s*\]\s*\([^)]*\)', '', candidate)
                candidate = re.sub(r'https?://\S+', '', candidate)
                candidate = re.sub(r'\s*>\s*', ' ', candidate)

                candidate = re.sub(r'\s+([;:,])', r'\1', candidate)
                candidate = re.sub(r'([;:])(?=\S)', r'\1 ', candidate)
                candidate = re.sub(r'(,)(?=[^\s\d])', r'\1 ', candidate)
                candidate = re.sub(r'[ \t]+', ' ', candidate)
                candidate = normalize_decimal_commas(candidate)
                candidate = fix_article_number_spacing(candidate)
                candidate = candidate.replace('\\', '')
                candidate = candidate.rstrip(',[]* ')
                candidate = collapse_duplicate_text(candidate)

                if candidate:
                    normalized_lines.append(candidate)

        if not normalized_lines and title_normalized:
            normalized_lines.append(title_normalized.strip())

        if normalized_lines:
            cleaned_text = "\n".join(normalized_lines)
            rows.append({"title": title_normalized or title, "text": cleaned_text})
            print(f"[VERBOSE] Item '{title_normalized or title}' extraido e limpo.")
    return [], "", cleaned_text

def find_and_convert_markdown_to_csv():
    """
    Função principal que localiza o .md, processa a estrutura e gera o .csv.
    """
    print("--- Conversor Estruturado de Markdown de Leis para CSV (Notion) ---")

    print("\n[INFO] Procurando por arquivo .md na pasta...")
    markdown_files = glob.glob('*.md')

    if not markdown_files:
        print("\n[ERRO] Nenhum arquivo .md foi encontrado. Coloque o arquivo de legislação na mesma pasta.")
        return

    input_filename = markdown_files[0]
    print(f"[INFO] Arquivo encontrado: '{input_filename}'")
    
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}.csv"

    print(f"\nIniciando a conversão estruturada para '{output_filename}'...")

    # Expressoes Regulares - correcao aplicada aqui
    art_pattern = re.compile(r'^(Art\.\s*\d+(?:\.\d+)*(?:[.\s]*(?:\u00ba|\u00aa|o))?(?:-[A-Z]+)?(?:\.\s*)?)(.*)')
    par_pattern = re.compile(
        r'^('
        r'\u00a7\s*\d+(?:-[A-Z]+)?[\w.]*\u00ba?'
        r'|Par\u00e1grafo \u00fanico'
        r')'
        r'(?:\s*[-–—:]\s*|\.\s*)?'
        r'(.*)'
    )
    inc_pattern = re.compile(r'^([IVXLCDM]+)\s*[-\u2013\u2014]\s*(.*)')
    ali_pattern = re.compile(r'^([a-z])\)(.*)')
    num_pattern = re.compile(r'^(\d+)[\.\)\-]\s*(.*)')
    header_pattern = re.compile(r'^\s*(PARTE|LIVRO|T\u00cdTULO|CAP\u00cdTULO|Se\u00e7\u00e3o|Subse\u00e7\u00e3o|ANEXO)\s', re.IGNORECASE)
    header_search_pattern = re.compile(r'\s*(PARTE|LIVRO|T\u00cdTULO|CAP\u00cdTULO|Se\u00e7\u00e3o|Subse\u00e7\u00e3o|ANEXO)\s', re.IGNORECASE)
    inline_art_pattern = re.compile(r'Art\.\s*\d+(?:\.\d+)*(?:[.\s]*(?:\u00ba|\u00aa|o))?(?:-[A-Z]+)?(?:\.\s*)?')
    inline_paragraph_split_pattern = re.compile(
        r'(?=(?:\u00a7\s*\d+(?:-[A-Z]+)?[\w.]*\u00ba?|Par\u00e1grafo \u00fanico)(?:\s*[-–—:]\s*|\s+))',
        re.IGNORECASE,
    )

    with open(input_filename, 'r', encoding='utf-8') as md_file:
        content = md_file.read()

    start_index = -1
    match = re.search(r"Art\.\s*1", content, re.IGNORECASE)
    if match:
        start_index = match.start()
    
    if start_index == -1:
        print("\n[ERRO] 'Art. 1.º' não foi encontrado no arquivo.")
        return
    
    content_from_articles = content[start_index:]

    all_rows = []
    current_text_lines = []
    current_title = ""
    context = {"art": "", "par": "", "inc": "", "ali": "", "num": ""}
    hierarchy_text = {"art": "", "par": "", "inc": "", "ali": "", "num": ""}
    current_level = None
    last_line_was_header_desc = False

    def get_parent_text(*levels):
        """Retorna o texto herdado mais específico disponível para os níveis informados."""
        for level in levels:
            parent_text = hierarchy_text.get(level)
            if parent_text:
                return parent_text

        fallback_parts = []
        order = ("art", "par", "inc", "ali", "num")
        for level in order:
            if level in levels and context.get(level):
                fallback_parts.append(context[level])

        return "\n".join(part for part in fallback_parts if part)

    for line in content_from_articles.splitlines():
        line = normalize_space_characters(line)
        original_line = line.strip()
        if not original_line:
            continue

        expanded_line = original_line
        expanded_line = re.sub(r';\s*(?=[IVXLCDM]+\s*[---])', ';\n', expanded_line)
        expanded_line = re.sub(r';\s*(?=[a-z]\))', ';\n', expanded_line)
        expanded_line = re.sub(r';\s*(?=\d+[\.\)\-])', ';\n', expanded_line)

        pending_lines = expanded_line.splitlines()

        while pending_lines:
            sub_line = pending_lines.pop(0)
            sub_line = normalize_space_characters(sub_line)
            line_stripped = sub_line.strip()
            if not line_stripped:
                continue

            line_stripped = re.sub(r'^#{1,6}\s*', '', line_stripped)
            line_stripped = re.sub(r'^[_*]+', '', line_stripped)
            line_stripped = re.sub(r'\[\s*\]\s*\([^)]+\)', '', line_stripped)
            line_stripped = re.sub(r'\[([^\]]+)\]', r'\1', line_stripped)
            line_stripped = re.sub(r'\(\s*(?:https?://|\.\.?/|#)[^)]+\)', '', line_stripped)
            line_stripped = re.sub(r'\(\s*[\w\u00C0-\u024F./#%-]*\.html?[^)]*\)', '', line_stripped, flags=re.IGNORECASE)
            line_stripped = re.sub(r'https?://\S+', '', line_stripped)
            line_stripped = normalize_ordinals(line_stripped)
            line_stripped = re.sub(r'\u00a7\s+', '\u00a7', line_stripped)
            line_stripped = re.sub(
                r'(\u00a7)([1-9])(?!\u00ba|\d)',
                lambda m: f"{m.group(1)}{m.group(2)}\u00ba",
                line_stripped,
            )
            line_stripped = re.sub(r'^>+\s*', '', line_stripped)
            if '>' in line_stripped:
                normalized = re.sub(r'(?:\s*>\s*){2,}', '\n', line_stripped)
                if normalized != line_stripped:
                    split_parts = normalized.splitlines()
                    if split_parts:
                        pending_lines = split_parts + pending_lines
                        continue
                    line_stripped = normalized
                line_stripped = re.sub(r'\s*>\s*', ' ', line_stripped)

            line_stripped = line_stripped.strip()

            if line_stripped:
                search_pos = 1
                while True:
                    split_match = inline_paragraph_split_pattern.search(line_stripped, search_pos)
                    if not split_match:
                        break
                    prefix = line_stripped[:split_match.start()]
                    trimmed_prefix = prefix.rstrip()
                    has_boundary = not trimmed_prefix or trimmed_prefix[-1] in ".;:–—-)]}"
                    if has_boundary:
                        before = prefix.rstrip(' -–—')
                        after = line_stripped[split_match.start():].lstrip(' -–—')
                        if after:
                            pending_lines.insert(0, after)
                        line_stripped = before.strip()
                        search_pos = 1
                        if not line_stripped:
                            break
                    else:
                        search_pos = split_match.start() + 1
                if not line_stripped:
                    continue

            matches = list(inline_art_pattern.finditer(line_stripped))
            if matches and matches[0].start() == 0:
                segments = []
                last_pos = 0
                for i, m in enumerate(matches):
                    if m.start() > last_pos:
                        seg = line_stripped[last_pos:m.start()].strip()
                        if seg:
                            segments.append(seg)
                    next_start = matches[i + 1].start() if i + 1 < len(matches) else len(line_stripped)
                    seg = line_stripped[m.start():next_start].strip()
                    if seg:
                        segments.append(seg)
                    last_pos = next_start
                if len(segments) > 1 or (segments and segments[0] != line_stripped):
                    pending_lines = segments + pending_lines
                    continue
                if segments:
                    line_stripped = segments[0]

            art_match = art_pattern.match(line_stripped)
            par_match = par_pattern.match(line_stripped)
            inc_match = inc_pattern.match(line_stripped)
            ali_match = ali_pattern.match(line_stripped)
            num_match = num_pattern.match(line_stripped)
            header_match = header_pattern.match(line_stripped)
            
            is_all_caps_desc = line_stripped.isupper() and len(line_stripped.split()) < 10

            if art_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                art_title = clean_title(art_match.group(1))
                context = {"art": art_title, "par": "", "inc": "", "ali": "", "num": ""}
                current_level = "art"
                current_title = context["art"]
                current_text_lines = [line_stripped]
                hierarchy_text["art"] = ""
                hierarchy_text["par"] = ""
                hierarchy_text["inc"] = ""
                hierarchy_text["ali"] = ""
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False
            
            elif par_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                par_title = clean_title(par_match.group(1))
                context["par"] = par_title
                context["inc"] = ""
                context["ali"] = ""
                context["num"] = ""
                current_level = "par"

                current_title = f'{context["art"]}, {context["par"]}'
                parent_text = get_parent_text("art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["inc"] = ""
                hierarchy_text["ali"] = ""
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif inc_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                context["inc"] = inc_match.group(1).strip()
                context["ali"] = ""
                context["num"] = ""
                current_level = "inc"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                title_parts.append(context["inc"])
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["ali"] = ""
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif ali_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                alinea = ali_match.group(1).strip() + ")"
                context["ali"] = alinea
                context["num"] = ""
                current_level = "ali"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                if context["inc"]:
                    title_parts.append(context["inc"])
                title_parts.append(alinea)
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("inc", "par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif num_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                numero = num_match.group(1).strip()
                context["num"] = numero
                current_level = "num"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                if context["inc"]:
                    title_parts.append(context["inc"])
                if context["ali"]:
                    title_parts.append(context["ali"])
                title_parts.append(numero)
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("ali", "inc", "par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                last_line_was_header_desc = False
                
            elif header_match or (last_line_was_header_desc and is_all_caps_desc):
                current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                print(f"[VERBOSE] Ignorando cabeçalho: '{line_stripped}'")
                last_line_was_header_desc = True
                current_title, current_text_lines = "", []
                current_level = None

            else:
                if current_title:
                    header_in_line_match = header_search_pattern.search(line_stripped)
                    if header_in_line_match:
                        header_value = header_in_line_match.group(1)
                        if not (header_value.isupper() or header_value.istitle()):
                            header_in_line_match = None
                    if header_in_line_match:
                        text_before_header = line_stripped[:header_in_line_match.start()].strip()
                        if text_before_header:
                            current_text_lines.append(text_before_header)
                        
                        current_text_lines, current_title, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
                        if cleaned_parent and current_level:
                            hierarchy_text[current_level] = cleaned_parent
                        
                        print(f"[VERBOSE] Ignorando cabeçalho no meio da linha: '{line_stripped[header_in_line_match.start():]}'")
                        last_line_was_header_desc = True
                        current_title, current_text_lines = "", []
                        current_level = None
                    else:
                        if current_text_lines:
                            current_text_lines[-1] = f"{current_text_lines[-1]} {line_stripped}"
                        else:
                            current_text_lines.append(line_stripped)
                        last_line_was_header_desc = False
                else:
                    last_line_was_header_desc = False
    _, _, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
    if cleaned_parent and current_level:
        hierarchy_text[current_level] = cleaned_parent

    footer_pattern = re.compile(
        r'(\b[A-Za-z?-?][^,\n\d]+,\s+\d{1,2}\s+de\s+[A-Za-z?-?]+\s+de\s+\d{4}.*$)',
        re.IGNORECASE | re.DOTALL,
    )
    if all_rows:
        last_text = all_rows[-1]['text']
        match_footer = footer_pattern.search(last_text)
        if match_footer and 'lei n' not in match_footer.group(1).lower():
            cleaned_last = last_text[:match_footer.start()].rstrip()
        else:
            cleaned_last = last_text.strip()
        all_rows[-1]['text'] = cleaned_last
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            writer.writerow(["ID", "Nome (Artigo)", "Texto do Artigo"])
            
            for idx, row in enumerate(all_rows, start=1):
                writer.writerow([idx, row['title'], row['text']])
        
        print("\n---------------------------------------------------------")
        if not all_rows:
            print("[AVISO] Nenhum artigo foi extraído. Verifique o formato do arquivo .md.")
        else:
            print(f"[SUCESSO] Conversão concluída! {len(all_rows)} itens foram processados.")
            print(f"O arquivo CSV foi salvo como: '{output_filename}'")
        print("---------------------------------------------------------")

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro inesperado ao escrever o arquivo CSV: {e}")

if __name__ == "__main__":
    find_and_convert_markdown_to_csv()
