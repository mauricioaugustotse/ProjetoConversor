# Diagnostico de padroes e candidatos a duplicata (somente leitura). Gera CSV mestre da RAIZ (exclui OneDrive TSE).
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$tse  = Join-Path $base 'OneDrive - TRIBUNAL SUPERIOR ELEITORAL'

Write-Output "=== Diretorios de 1o nivel em HD_Mau ==="
Get-ChildItem -LiteralPath $base -Directory -Force | ForEach-Object {
  $n = (Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object).Count
  Write-Output ("  [DIR] {0,7:N0} arq  {1}" -f $n, $_.Name)
}
$rootFilesCount = (Get-ChildItem -LiteralPath $base -File -Force | Measure-Object).Count
Write-Output ("  (arquivos soltos no 1o nivel da raiz: {0:N0})" -f $rootFilesCount)

# Coleta arquivos da RAIZ (tudo menos a pasta TSE)
Write-Output "`n=== Coletando indice mestre (exceto OneDrive TSE) ==="
$prefixLen = $base.Length + 1
$files = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notlike "$tse*" } |
  Select-Object @{n='Rel';e={$_.FullName.Substring($prefixLen)}}, Name,
                @{n='Ext';e={$_.Extension.ToLower()}}, Length,
                @{n='Modified';e={$_.LastWriteTime.ToString('yyyy-MM-dd')}}, FullName
$files | Export-Csv "$out\indice_mestre_raiz.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Arquivos no indice mestre (raiz): {0:N0}" -f $files.Count)

# Padroes de nome
$reNum  = '^\(\d+\)'                          # comeca com (N)
$reCopy = '(\(\d+\)\.[A-Za-z0-9]+$)|(- C[oó]pia)|(- Copy)|(C[oó]pia de )|(\bcopy\b)'
$cNum  = ($files | Where-Object { $_.Name -match $reNum }).Count
$cCopy = ($files | Where-Object { $_.Name -match $reCopy }).Count
$cDash = ($files | Where-Object { $_.Name -like '* - *' }).Count
Write-Output "`n=== Padroes de nome ==="
Write-Output ("  Comecam com '(N)'            : {0:N0}" -f $cNum)
Write-Output ("  Marcadores de copia no nome  : {0:N0}" -f $cCopy)
Write-Output ("  Contem ' - ' (estruturado)   : {0:N0}" -f $cDash)

# Candidatos a duplicata por (Nome + Tamanho) identicos
Write-Output "`n=== Candidatos a duplicata ==="
$byNameSize = $files | Group-Object { "$($_.Name)|$($_.Length)" } | Where-Object Count -gt 1
$dupNameSizeFiles = ($byNameSize | Measure-Object Count -Sum).Sum
$dupNameSizeExtra = $dupNameSizeFiles - $byNameSize.Count
Write-Output ("  Grupos com mesmo NOME+TAMANHO : {0:N0}  (arquivos: {1:N0}; excedentes: {2:N0})" -f $byNameSize.Count, $dupNameSizeFiles, $dupNameSizeExtra)

# Candidatos por TAMANHO identico (>0 bytes) - base para hashing posterior
$bySize = $files | Where-Object { $_.Length -gt 0 } | Group-Object Length | Where-Object Count -gt 1
$cand = ($bySize | Measure-Object Count -Sum).Sum
[long]$candBytes = ($bySize | ForEach-Object { $_.Group | Select-Object -Skip 1 } | Measure-Object Length -Sum).Sum
Write-Output ("  Grupos com mesmo TAMANHO      : {0:N0}  (arquivos candidatos: {1:N0})" -f $bySize.Count, $cand)
Write-Output ("  Espaco potencialmente recuperavel (se forem identicos): {0:N2} GB" -f ($candBytes/1GB))
Write-Output "`nCSV mestre salvo em $out\indice_mestre_raiz.csv"