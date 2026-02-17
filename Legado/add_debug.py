from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
marker = "                candidate = collapse_duplicate_text(candidate)\n\n                if candidate:\n"
replacement = "                candidate = collapse_duplicate_text(candidate)\n                if candidate:\n                    if title and title.startswith('Art. 3'):\n                        print('DEBUG candidate', repr(candidate))\n"
if marker not in text:
    raise SystemExit('marker not found')
text = text.replace(marker, replacement, 1)
path.write_text(text, encoding='utf-8')
