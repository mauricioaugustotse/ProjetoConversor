#!/bin/bash
# Extracao final: PDF/DOCX/RTF/TXT faltantes. DOCX cortado com head -c (evita sed travar em XML gigante).
# 16 workers em loop (sem spawn de bash por arquivo). Idempotente.
export LC_ALL=C
BASE="/c/Users/mauri/HD_Mau"
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
export OUT

cut -f1 "$OUT/extraido.tsv" | sort -u > "$OUT/done.txt"
find "$BASE" -type f \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.rtf' -o -iname '*.txt' \) | sort -u > "$OUT/all_rapido.txt"
comm -23 "$OUT/all_rapido.txt" "$OUT/done.txt" > "$OUT/todo_rapido.txt"
N=$(wc -l < "$OUT/todo_rapido.txt"); echo "Faltam: $N"
[ "$N" -eq 0 ] && { echo "Nada a fazer."; exit 0; }

rm -rf "$OUT/partes"; mkdir -p "$OUT/partes"
split -n l/16 "$OUT/todo_rapido.txt" "$OUT/partes/p_" 2>/dev/null || split -l $(( (N+15)/16 )) "$OUT/todo_rapido.txt" "$OUT/partes/p_"

worker(){
  while IFS= read -r f; do
    low="${f,,}"; txt=""
    case "$low" in
      *.pdf)  txt=$(timeout -s KILL 15 pdftotext -enc UTF-8 -f 1 -l 1 "$f" - 2>/dev/null) ;;
      *.docx) txt=$(timeout -s KILL 15 bash -c 'unzip -p "$1" word/document.xml 2>/dev/null | head -c 30000 | sed "s/<[^>]*>/ /g"' _ "$f") ;;
      *.rtf)  txt=$(head -c 40000 "$f" 2>/dev/null | sed -e 's/\\[a-zA-Z]\+[0-9]*//g' -e 's/[{}]//g') ;;
      *.txt)  txt=$(head -c 6000 "$f" 2>/dev/null) ;;
    esac
    clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
    printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
  done < "$1"
}
for p in "$OUT"/partes/p_*; do worker "$p" & done
wait
echo "FIM. Total no tsv: $(wc -l < "$OUT/extraido.tsv")"
