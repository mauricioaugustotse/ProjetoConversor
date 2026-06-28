#!/bin/bash
export LC_ALL=C
A="/c/Users/mauri/OneDrive/Documentos/06 - Estudos, concursos e leitura/TESE"
B="/c/Users/mauri/OneDrive/Documentos/05 - Jurídico e trabalho/TSE/ONE DRIVE/Documentos/TESE"
OUT="/c/Users/mauri/ProjetoConversor"
echo "Hasheando TESE atual (A)..."
find "$A" -type f -exec md5sum {} + 2>/dev/null | awk '{print $1}' | sort > "$OUT/hash_tese_A.txt"
echo "Hasheando TESE backup (B)..."
find "$B" -type f -exec md5sum {} + 2>/dev/null > "$OUT/hash_tese_B_full.txt"
awk '{print $1}' "$OUT/hash_tese_B_full.txt" | sort > "$OUT/hash_tese_B.txt"
nA=$(wc -l < "$OUT/hash_tese_A.txt"); nB=$(wc -l < "$OUT/hash_tese_B.txt")
# quantos hashes de B existem em A
comum=$(comm -12 "$OUT/hash_tese_A.txt" "$OUT/hash_tese_B.txt" | wc -l)
# arquivos de B cujo hash NAO esta em A (unicos no backup)
echo "TESE atual (A): $nA arquivos | TESE backup (B): $nB arquivos"
echo "Hashes de B presentes em A (duplicados): $comum"
echo "Arquivos de B unicos (nao estao em A): $((nB - comum))"
