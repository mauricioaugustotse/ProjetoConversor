from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
lines = path.read_text(encoding='utf-8').splitlines()
for idx,line in enumerate(lines):
    if line.strip().startswith('header_search_pattern = re.compile'):
        lines.insert(idx+1, '    inline_art_pattern = re.compile(r"Art\\.\\s*\\d+(?:\\.\\d+)*(?:-[A-Z]+)?(?:\\s*(?:�|�|o))?")')
        break
path.write_text('\n'.join(lines)+"\n", encoding='utf-8')
