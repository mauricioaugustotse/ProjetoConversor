# Gera lotes dos arquivos de 05-Diversos COM texto, com trecho do conteudo, para classificacao por IA.
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$tsv = "$out\conteudo\extraido_full.tsv"
$loteDir = "$out\lotes_div"; New-Item -ItemType Directory -Path $loteDir -Force | Out-Null
Get-ChildItem $loteDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
$outDir = "$out\lotes_div_out"; New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Get-ChildItem $outDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$map = New-Object System.Collections.Generic.List[object]
$buf = New-Object System.Collections.Generic.List[string]
$id=0; $li=0; $size=40; $comTexto=0
foreach($line in [IO.File]::ReadLines($tsv, [Text.Encoding]::UTF8)){
  $p = $line -split "`t", 3
  if($p.Count -lt 3){ continue }
  $path=$p[0]; $len=[int]$p[1]; $txt=$p[2]
  if($path -notmatch '/05 - Diversos'){ continue }
  if($len -lt 80){ continue }
  $win = ($path -replace '^/c/','C:/' -replace '^/C/','C:/') -replace '/','\'
  if(-not (Test-Path -LiteralPath $win)){ continue }   # ja movido / nao existe mais em Diversos
  $comTexto++
  $comp = ($txt -replace '\s','')
  $cnj=''; $m=[regex]::Match($comp,'\d{7}-?\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'); if($m.Success){ $cnj=$m.Value }
  $id++
  $trecho = if($txt.Length -gt 700){ $txt.Substring(0,700) } else { $txt }
  $buf.Add(("{0}`t{1}" -f $id, $trecho))
  $map.Add([pscustomobject]@{ ID=$id; Path=$win; CNJ=$cnj; Ext=[IO.Path]::GetExtension($win); NomeAtual=[IO.Path]::GetFileName($win) })
  if($buf.Count -ge $size){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $buf.Clear(); $li++ }
}
if($buf.Count){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $li++ }
$map | Export-Csv "$out\map_div.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Diversos com texto: {0} | lotes: {1} | com CNJ: {2}" -f $comTexto, $li, @($map | Where-Object { $_.CNJ -ne '' }).Count)