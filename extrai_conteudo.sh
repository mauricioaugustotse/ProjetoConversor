#!/bin/bash
# Extrai a 1a pagina/inicio de cada documento (PDF/DOC/DOCX/RTF/TXT) localmente, sem IA.
# Saida: extraido.tsv  ->  CAMINHO <TAB> LEN <TAB> TEXTO(ate 1200 chars, 1 linha)
BASE="/c/Users/mauri/HD_Mau"
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
mkdir -p "$OUT"
: > "$OUT/extraido.tsv"
export OUT

extrai() {
  f="$1"
  low="${f,,}"
  txt=""
  case "$low" in
    *.pdf)  txt=$(pdftotext -enc UTF-8 -f 1 -l 1 "$f" - 2>/dev/null) ;;
    *.docx) txt=$(unzip -p "$f" word/document.xml 2>/dev/null | sed 's/<[^>]*>/ /g') ;;
    *.doc)  txt=$(antiword "$f" 2>/dev/null) ;;
    *.rtf)  txt=$(sed -e 's/\\[a-zA-Z]\+[0-9]*//g' -e 's/[{}]//g' "$f" 2>/dev/null) ;;
    *.txt)  txt=$(head -c 6000 "$f" 2>/dev/null) ;;
  esac
  clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
  len=${#clean}
  printf '%s\t%s\t%s\n' "$f" "$len" "$clean" >> "$OUT/extraido.tsv"
}
export -f extrai

find "$BASE" -type f \( -iname '*.pdf' -o -iname '*.doc' -o -iname '*.docx' -o -iname '*.rtf' -o -iname '*.txt' \) -print0 \
  | xargs -0 -P 12 -I{} bash -c 'extrai "$@"' _ {}

echo "Linhas extraidas: $(wc -l < "$OUT/extraido.tsv")"
