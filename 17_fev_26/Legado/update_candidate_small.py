from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
old = "                candidate = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1�', candidate)\n                candidate = re.sub(r'�\\s+', '�', candidate)\n"
new = "                candidate = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1�', candidate)\n                candidate = re.sub(r'�\\s+', '�', candidate)\n                candidate = re.sub(r'(?i)(Lei)\\s+n\\s*o\\b', r'\\1 n�', candidate)\n"
if old not in text:
    raise SystemExit('pattern not found (� version)')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
