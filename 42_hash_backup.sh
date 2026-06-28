#!/bin/bash
export LC_ALL=C
BASE="/c/Users/mauri/OneDrive/Documentos"
BACKUP="$BASE/05 - Jurídico e trabalho/TSE/ONE DRIVE"
OUT="/c/Users/mauri/ProjetoConversor"
echo "1/3 Hasheando tudo FORA do backup (pode demorar)..."
find "$BASE" -type f -not -path "$BACKUP/*" -not -path "*/LUIZ CELSO VIEIRA/*" -exec md5sum {} + 2>/dev/null > "$OUT/resto.md5"
awk '{print substr($0,1,32)}' "$OUT/resto.md5" | sort -u > "$OUT/resto_hashes.txt"
echo "2/3 Hasheando o backup ONE DRIVE..."
find "$BACKUP" -type f -exec md5sum {} + 2>/dev/null > "$OUT/backup.md5"
echo "3/3 Separando duplicatas x unicos..."
: > "$OUT/backup_dup.txt"; : > "$OUT/backup_uniq.txt"
awk 'NR==FNR{h[$1]=1; next}{hash=substr($0,1,32); path=substr($0,35); if(hash in h) print path >> "'"$OUT"'/backup_dup.txt"; else print path >> "'"$OUT"'/backup_uniq.txt"}' "$OUT/resto_hashes.txt" "$OUT/backup.md5"
echo "RESULTADO:"
echo "  Backup total: $(wc -l < "$OUT/backup.md5")"
echo "  Duplicatas (existem fora do backup): $(wc -l < "$OUT/backup_dup.txt")"
echo "  Unicos (so existem no backup): $(wc -l < "$OUT/backup_uniq.txt")"
echo "  -- unicos por pasta de 1o/2o nivel dentro do backup --"
sed "s#^$BACKUP/##" "$OUT/backup_uniq.txt" | awk -F/ '{print $1 (NF>1?"/"$2:"")}' | sort | uniq -c | sort -rn | head -20
