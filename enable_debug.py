from pathlib import Path
path = Path('LEGISLACAO_md_to_csv_converter.py')
text = path.read_text(encoding='utf-8')
text = text.replace('# print('CANDIDATE', repr(candidate))', "print('CANDIDATE', repr(candidate))")
path.write_text(text, encoding='utf-8')
