# Gera indice do estado atual com Nat/Tipo derivados do caminho. Somente metadados.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$prefixLen = $base.Length + 1
$rows = New-Object System.Collections.Generic.List[object]
$files = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '00 - INDICE.md' }
foreach($f in $files){
  $rel = $f.FullName.Substring($prefixLen)
  $seg = $rel -split '\\'
  $nat = $seg[0]
  $tipo = ''
  if($nat -like '01 - Jur*' -and $seg.Count -ge 3){ $tipo = $seg[1] }
  $rows.Add([pscustomobject]@{ Rel=$rel; Name=$f.Name; Ext=$f.Extension.ToLower(); Length=[long]$f.Length; Nat=$nat; Tipo=$tipo; FullName=$f.FullName })
}
$rows | Export-Csv "$out\indice_atual.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Indice atual: {0:N0} arquivos" -f $rows.Count)
$rows | Group-Object Nat | Sort-Object Name | ForEach-Object { Write-Output ("  {0,7:N0}  {1}" -f $_.Count, $_.Name) }