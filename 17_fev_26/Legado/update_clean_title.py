from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
old = "    cleaned = cleaned.rstrip('.')\n    cleaned = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1�', cleaned)\n    cleaned = re.sub(r'�\\s+', r'�', cleaned)\n"
new = "    cleaned = cleaned.rstrip('.')\n    cleaned = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1�', cleaned)\n    cleaned = re.sub(r'�\\s+', r'�', cleaned)\n    cleaned = re.sub(r'(?i)(Lei)\\s+n\\s*o\\b', r'\\1 nº', cleaned)\n"
if old not in text:
    raise SystemExit('pattern not found for clean_title')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
