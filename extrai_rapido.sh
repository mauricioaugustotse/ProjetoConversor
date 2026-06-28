#!/bin/bash
# Extrai apenas os formatos rapidos faltantes (PDF/DOCX/RTF/TXT), pulando .doc (antiword inviavel em escala).
export LC_ALL=C
BASE="/c/Users/mauri/HD_Mau"
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
export OUT

cut -f1 "$OUT/extraido.tsv" | sort -u > "$OUT/done.txt"
find "$BASE" -type f \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.rtf' -o -iname '*.txt' \) | sort -u > "$OUT/all_rapido.txt"
comm -23 "$OUT/all_rapido.txt" "$OUT/done.txt" > "$OUT/todo_rapido.txt"
echo "Faltam (PDF/DOCX/RTF/TXT): $(wc -l < "$OUT/todo_rapido.txt")"

extrai() {
  f="$1"; low="${f,,}"; txt=""
  case "$low" in
    *.pdf)  txt=$(timeout -s KILL 15 pdftotext -enc UTF-8 -f 1 -l 1 "$f" - 2>/dev/null) ;;
    *.docx) txt=$(timeout -s KILL 15 unzip -p "$f" word/document.xml 2>/dev/null | sed 's/<[^>]*>/ /g') ;;
    *.rtf)  txt=$(timeout -s KILL 6 sed -e 's/\\[a-zA-Z]\+[0-9]*//g' -e 's/[{}]//g' "$f" 2>/dev/null) ;;
    *.txt)  txt=$(head -c 6000 "$f" 2>/dev/null) ;;
  esac
  clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
  printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
}
export -f extrai
xargs -d'\n' -a "$OUT/todo_rapido.txt" -P 16 -I{} bash -c 'extrai "$@"' _ {}
echo "FIM. Total no tsv: $(wc -l < "$OUT/extraido.tsv")"
