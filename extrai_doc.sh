#!/bin/bash
# Extrai os .doc faltantes via antiword com SIGKILL no timeout (mata travamentos de verdade).
OUT="/c/Users/mauri/ProjetoConversor/conteudo"
export OUT
extraidoc(){
  f="$1"
  txt=$(timeout -s KILL 6 antiword "$f" 2>/dev/null)
  clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
  printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/extraido.tsv"
}
export -f extraidoc
grep -iE '\.doc$' "$OUT/todo2.txt" | xargs -d'\n' -P 16 -I{} bash -c 'extraidoc "$@"' _ {}
echo "DOCs processados. Total no tsv: $(wc -l < "$OUT/extraido.tsv")"
