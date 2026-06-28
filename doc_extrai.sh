#!/bin/bash
# Extrai 1a pagina/inicio dos textuais de Documentos (exclui LUIZ CELSO VIEIRA). 16 workers em loop.
export LC_ALL=C
BASE="/c/Users/mauri/OneDrive/Documentos"
OUT="/c/Users/mauri/ProjetoConversor/conteudo_doc"
mkdir -p "$OUT"
export OUT
find "$BASE" -type f \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.doc' -o -iname '*.rtf' -o -iname '*.txt' \) \
  | grep -v '/LUIZ CELSO VIEIRA/' | sort -u > "$OUT/all.txt"
# idempotente: pular ja extraidos
touch "$OUT/extraido.tsv"
cut -f1 "$OUT/extraido.tsv" | sort -u > "$OUT/done.txt"
comm -23 "$OUT/all.txt" "$OUT/done.txt" > "$OUT/todo.txt"
N=$(wc -l < "$OUT/todo.txt"); echo "Total: $(wc -l < "$OUT/all.txt") | faltam: $N"
[ "$N" -eq 0 ] && { echo "Nada a fazer."; exit 0; }
rm -rf "$OUT/partes"; mkdir -p "$OUT/partes"
split -n l/16 "$OUT/todo.txt" "$OUT/partes/p_" 2>/dev/null || split -l $(( (N+15)/16 )) "$OUT/todo.txt" "$OUT/partes/p_"
worker(){
  while IFS= read -r f; do
    low="${f,,}"; txt=""
    case "$low" in
      *.pdf)  txt=$(timeout -s KILL 20 pdftotext -enc UTF-8 -f 1 -l 1 "$f" - 2>/dev/null) ;;
      *.docx) txt=$(timeout -s KILL 20 bash -c 'unzip -p "$1" word/document.xml 2>/dev/null | head -c 30000 | sed "s/<[^>]*>/ /g"' _ "$f") ;;
      *.doc)  txt=$(timeout -s KILL 8 antiword "$f" 2>/dev/null) ;;
      *.rtf)  txt=$(head -c 40000 "$f" 2>/dev/null | sed -e 's/\\[a-zA-Z]\+[0-9]*//g' -e 's/[{}]//g') ;;
      *.txt)  txt=$(head -c 6000 "$f" 2>/dev/null) ;;
    esac
    clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1000)
    printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
  done < "$1"
}
for p in "$OUT"/partes/p_*; do worker "$p" & done
wait
echo "FIM. Linhas: $(wc -l < "$OUT/extraido.tsv") | com texto util: $(awk -F'\t' '$2>=60' "$OUT/extraido.tsv"|wc -l)"
