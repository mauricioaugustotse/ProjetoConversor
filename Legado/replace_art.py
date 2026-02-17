from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
text = text.replace('    art_pattern = re.compile(r"^(Art\\.\\s*\\d+(?:-[A-Z]+)?(?:\\.\\s*)?(?:�|�|o)?)(.*)")\n', '    art_pattern = re.compile(r"^(Art\\.\\s*\\d+(?:\\.\\d+)*(?:-[A-Z]+)?(?:\\s*(?:�|�|o))?)(.*)")\n')
path.write_text(text, encoding='utf-8')
