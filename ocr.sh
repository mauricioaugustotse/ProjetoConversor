#!/bin/bash
# OCR dos PDFs escaneados (sem texto). pdftoppm (1a pagina -> PNG) + tesseract -l por. 16 workers.
export LC_ALL=C
TES="/c/Program Files/Tesseract-OCR/tesseract.exe"
PPM="/c/Users/mauri/AppData/Local/Microsoft/WinGet/Packages/oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe/poppler-25.07.0/Library/bin/pdftoppm.exe"
TDATA="/c/Users/mauri/ProjetoConversor/tessdata"
OUT="/c/Users/mauri/ProjetoConversor/conteudo"

awk -F'\t' '$2<80 && tolower($1) ~ /\.pdf$/ {print $1}' "$OUT/extraido.tsv" | sort -u > "$OUT/ocr_todo.txt"
N=$(wc -l < "$OUT/ocr_todo.txt"); echo "Escaneados p/ OCR: $N"
: > "$OUT/ocr.tsv"
rm -rf "$OUT/partes_ocr"; mkdir -p "$OUT/partes_ocr"
split -n l/16 "$OUT/ocr_todo.txt" "$OUT/partes_ocr/p_" 2>/dev/null || split -l $(( (N+15)/16 )) "$OUT/ocr_todo.txt" "$OUT/partes_ocr/p_"

worker(){
  local wid="$1"; local lst="$2"; local tmp="/tmp/ocr_w${wid}"
  while IFS= read -r f; do
    rm -f ${tmp}*.png
    timeout -s KILL 30 "$PPM" -png -r 200 -f 1 -l 1 "$f" "$tmp" 2>/dev/null
    img="${tmp}-1.png"; [ -f "$img" ] || img=$(ls ${tmp}*.png 2>/dev/null | head -1)
    txt=""
    [ -f "$img" ] && txt=$(timeout -s KILL 30 "$TES" "$img" stdout -l por --tessdata-dir "$TDATA" 2>/dev/null)
    rm -f ${tmp}*.png
    clean=$(printf '%s' "$txt" | tr '\t\r\n' '   ' | tr -s ' ' | cut -c1-1200)
    printf '%s\t%s\t%s\n' "$f" "${#clean}" "$clean" >> "$OUT/ocr.tsv"
  done < "$lst"
}
i=0
for p in "$OUT"/partes_ocr/p_*; do worker "$i" "$p" & i=$((i+1)); done
wait
echo "FIM OCR. Linhas: $(wc -l < "$OUT/ocr.tsv") | com texto util: $(awk -F'\t' '$2>=80' "$OUT/ocr.tsv" | wc -l)"
