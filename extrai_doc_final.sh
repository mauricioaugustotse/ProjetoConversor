#!/bin/bash
# Extrai os .doc faltantes via antiword (timeout SIGKILL), 16 workers em loop. Idempotente.
export LC_ALL=C
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
export OUT
find "/c/Users/mauri/HD_Mau" -iname '*.doc' | sort -u > "$OUT/all_doc.txt"
cut -f1 "$OUT/extraido.tsv" | grep -iE '\.doc$' | sort -u > "$OUT/done_doc.txt"
comm -23 "$OUT/all_doc.txt" "$OUT/done_doc.txt" > "$OUT/todo_doc.txt"
N=$(wc -l < "$OUT/todo_doc.txt"); echo "Faltam .doc: $N"
[ "$N" -eq 0 ] && { echo "Nada a fazer."; exit 0; }
rm -rf "$OUT/partes_doc"; mkdir -p "$OUT/partes_doc"
split -n l/16 "$OUT/todo_doc.txt" "$OUT/partes_doc/p_" 2>/dev/null || split -l $(( (N+15)/16 )) "$OUT/todo_doc.txt" "$OUT/partes_doc/p_"
worker(){
  while IFS= read -r f; do
    txt=$(timeout -s KILL 6 antiword "$f" 2>/dev/null)
    clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
    printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
  done < "$1"
}
for p in "$OUT"/partes_doc/p_*; do worker "$p" & done
wait
echo "FIM .doc. Total no tsv: $(wc -l < "$OUT/extraido.tsv")"
