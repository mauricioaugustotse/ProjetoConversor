from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
old = "                candidate = candidate.replace('.�', '�').replace('.�', '�')\n                candidate = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1�', candidate)\n                candidate = re.sub(r'�\\s+', '�', candidate)\n                candidate = re.sub(r'\\s*\\(\\s*', ' (', candidate)\n"
new = "                candidate = candidate.replace('.º', 'º').replace('.ª', 'ª')\n                candidate = re.sub(r'(\\d+)\\s*[oO]\\b', r'\\1º', candidate)\n                candidate = re.sub(r'§\\s+', '§', candidate)\n                candidate = re.sub(r'(?i)(Lei)\\s+n\\s*o\\b', r'\\1 nº', candidate)\n                candidate = re.sub(r'\\s*\\(\\s*', ' (', candidate)\n"
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
