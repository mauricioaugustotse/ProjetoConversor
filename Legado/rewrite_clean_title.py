from pathlib import Path
text = Path('LEGISLACAO_md_to_csv_converter.py').read_text(encoding='utf-8')
start = text.index('def clean_title')
end = text.index('def collapse_duplicate_text')
new_block = "def clean_title(raw_title):\n    \"\"\"Função centralizada para limpar os títulos dos dispositivos.\"\"\"\n    cleaned = raw_title.strip().replace('.º', 'º').replace('.ª', 'ª')\n    cleaned = cleaned.rstrip('.')\n    cleaned = re.sub(r'(\d+)\s*[oO]\b', r'\\1º', cleaned)\n    cleaned = re.sub(r'§\\s+', r'§', cleaned)\n    cleaned = re.sub(r'(?i)(Lei)\\s+n\\s*o\\b', r'\\1 nº', cleaned)\n    return cleaned\n\n"
text = text[:start] + new_block + text[end:]
Path('LEGISLACAO_md_to_csv_converter.py').write_text(text, encoding='utf-8')
