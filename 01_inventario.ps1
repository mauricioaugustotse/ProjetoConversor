# Inventario de C:\Users\mauri\HD_Mau  (somente leitura de metadados; nao move/apaga/baixa nada)
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
New-Item -ItemType Directory -Path $out -Force | Out-Null

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$totFiles = 0; [long]$totBytes = 0
$byExt = @{}; $byTop = @{}; $byYear = @{}
$prefixLen = $base.Length + 1

Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
  $totFiles++
  $len = [long]$_.Length; $totBytes += $len

  $ext = $_.Extension.ToLower(); if([string]::IsNullOrEmpty($ext)){ $ext = '(sem ext)' }
  if($byExt.ContainsKey($ext)){ $byExt[$ext].Count++; $byExt[$ext].Bytes += $len }
  else { $byExt[$ext] = [pscustomobject]@{Ext=$ext; Count=[int]1; Bytes=$len} }

  $rel = $_.FullName.Substring($prefixLen)
  if($rel -match '\\'){ $top = ($rel -split '\\')[0] } else { $top = '(raiz)' }
  if($byTop.ContainsKey($top)){ $byTop[$top].Count++; $byTop[$top].Bytes += $len }
  else { $byTop[$top] = [pscustomobject]@{Pasta=$top; Count=[int]1; Bytes=$len} }

  $y = $_.LastWriteTime.Year
  if($byYear.ContainsKey($y)){ $byYear[$y]++ } else { $byYear[$y] = 1 }
}
$sw.Stop()

$totDirs = (Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue | Measure-Object).Count

# Exporta CSVs
$byExt.Values  | Sort-Object Bytes -Descending | Export-Csv "$out\inv_por_extensao.csv" -NoTypeInformation -Encoding UTF8
$byTop.Values  | Sort-Object Bytes -Descending | Export-Csv "$out\inv_por_pasta_top.csv" -NoTypeInformation -Encoding UTF8
$byYear.GetEnumerator() | Sort-Object Name | Select-Object @{n='Ano';e={$_.Name}}, @{n='Arquivos';e={$_.Value}} | Export-Csv "$out\inv_por_ano.csv" -NoTypeInformation -Encoding UTF8

function FmtGB([long]$b){ "{0:N2} GB" -f ($b/1GB) }

# Resumo na tela
Write-Output "===== RESUMO HD_Mau ====="
Write-Output ("Arquivos: {0:N0}   Pastas: {1:N0}   Tamanho total: {2}   Tempo: {3:N0}s" -f $totFiles, $totDirs, (FmtGB $totBytes), $sw.Elapsed.TotalSeconds)
Write-Output "`n--- TOP 15 pastas de 1o nivel (por tamanho) ---"
$byTop.Values | Sort-Object Bytes -Descending | Select-Object -First 15 | ForEach-Object { Write-Output ("  {0,10:N0}  {1,9}  {2}" -f $_.Count, (FmtGB $_.Bytes), $_.Pasta) }
Write-Output "`n--- TOP 25 extensoes (por tamanho) ---"
$byExt.Values | Sort-Object Bytes -Descending | Select-Object -First 25 | ForEach-Object { Write-Output ("  {0,10:N0}  {1,9}  {2}" -f $_.Count, (FmtGB $_.Bytes), $_.Ext) }
Write-Output "`n--- Arquivos por ano de modificacao ---"
$byYear.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Output ("  {0}: {1:N0}" -f $_.Name, $_.Value) }
Write-Output "`nCSVs salvos em $out"