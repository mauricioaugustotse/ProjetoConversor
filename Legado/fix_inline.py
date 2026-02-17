from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
lines = path.read_text(encoding='utf-8').splitlines()
lines = [line for line in lines if 'inline_art_pattern =' not in line]
for idx, line in enumerate(lines):
    if 'header_search_pattern' in line:
        lines.insert(idx + 1, '    inline_art_pattern = re.compile(r"Art\\.\\s*\\d+(?:\\.\\d+)*(?:-[A-Z]+)?(?:\\s*(?:º|°|o))?")')
        break
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
