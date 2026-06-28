# Gera lotes para a classificacao definitiva (materia + nome novo). Cada linha: ID<TAB>NAT<TAB>TIPO<TAB>NOME
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$loteDir = "$out\lotes2"
New-Item -ItemType Directory -Path $loteDir -Force | Out-Null
Get-ChildItem -LiteralPath $loteDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
$outDir = "$out\lotes2_out"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Get-ChildItem -LiteralPath $outDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$idx = Import-Csv "$out\indice_atual.csv" -Encoding UTF8
$size = 250; $id = 0; $li = 0
$map = New-Object System.Collections.Generic.List[object]
$buf = New-Object System.Collections.Generic.List[string]
foreach($f in $idx){
  $id++
  $buf.Add(("{0}`t{1}`t{2}`t{3}" -f $id, $f.Nat, $f.Tipo, $f.Name))
  $map.Add([pscustomobject]@{ ID=$id; FullName=$f.FullName; Nat=$f.Nat; Tipo=$f.Tipo; Name=$f.Name; Ext=$f.Ext })
  if($buf.Count -ge $size){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $buf.Clear(); $li++ }
}
if($buf.Count -gt 0){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $li++ }
$map | Export-Csv "$out\map2.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Lotes: {0} | Itens: {1:N0}" -f $li, $id)