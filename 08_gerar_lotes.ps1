# Gera lotes de nomes (somente os classificados como Diversos) para classificacao por IA.
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$loteDir = "$out\lotes"
New-Item -ItemType Directory -Path $loteDir -Force | Out-Null
Get-ChildItem -LiteralPath $loteDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$plan = Import-Csv "$out\plano_organizacao.csv" -Encoding UTF8
$div  = $plan | Where-Object { $_.Nat -like '05*' }

$size = 250; $id = 0; $loteIdx = 0
$map = New-Object System.Collections.Generic.List[object]
$buf = New-Object System.Collections.Generic.List[string]
foreach($d in $div){
  $id++
  $nome = Split-Path $d.Origem -Leaf
  $buf.Add(("{0}`t{1}" -f $id, $nome))
  $map.Add([pscustomobject]@{ ID=$id; Origem=$d.Origem })
  if($buf.Count -ge $size){
    Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir, $loteIdx) -Value $buf -Encoding UTF8
    $buf.Clear(); $loteIdx++
  }
}
if($buf.Count -gt 0){
  Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir, $loteIdx) -Value $buf -Encoding UTF8
  $loteIdx++
}
$map | Export-Csv "$out\diversos_map.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Lotes gerados: {0}  |  Itens: {1:N0}  |  Dir: {2}" -f $loteIdx, $id, $loteDir)