#!/bin/bash
# Reprocessa apenas os arquivos ainda nao extraidos, com timeout por arquivo (mata travamentos do antiword/pdftotext).
BASE="/c/Users/mauri/HD_Mau"
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
export OUT

# 1) lista do que ja foi extraido e do total; calcula faltantes
cut -f1 "$OUT/extraido.tsv" | sort -u > "$OUT/done.txt"
find "$BASE" -type f \( -iname '*.pdf' -o -iname '*.doc' -o -iname '*.docx' -o -iname '*.rtf' -o -iname '*.txt' \) | sort -u > "$OUT/all.txt"
comm -23 "$OUT/all.txt" "$OUT/done.txt" > "$OUT/todo.txt"
echo "Ja extraidos: $(wc -l < "$OUT/done.txt") | Total: $(wc -l < "$OUT/all.txt") | Faltam: $(wc -l < "$OUT/todo.txt")"

extrai() {
  f="$1"
  low="${f,,}"
  txt=""
  case "$low" in
    *.pdf)  txt=$(timeout 12 pdftotext -enc UTF-8 -f 1 -l 1 "$f" - 2>/dev/null) ;;
    *.docx) txt=$(timeout 12 unzip -p "$f" word/document.xml 2>/dev/null | sed 's/<[^>]*>/ /g') ;;
    *.doc)  txt=$(timeout 8 antiword "$f" 2>/dev/null) ;;
    *.rtf)  txt=$(timeout 8 sed -e 's/\\[a-zA-Z]\+[0-9]*//g' -e 's/[{}]//g' "$f" 2>/dev/null) ;;
    *.txt)  txt=$(head -c 6000 "$f" 2>/dev/null) ;;
  esac
  clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
  printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
}
export -f extrai

xargs -d'\n' -a "$OUT/todo.txt" -P 16 -I{} bash -c 'extrai "$@"' _ {}
echo "Total apos reprocessar: $(wc -l < "$OUT/extraido.tsv")"
