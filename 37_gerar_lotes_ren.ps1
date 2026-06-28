# Gera lotes para renomeacao por conteudo (Documentos). Cada linha: ID<TAB>PASTA<TAB>NOME_ATUAL<TAB>TRECHO
$ErrorActionPreference='SilentlyContinue'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$tsv = if($args[0]){ $args[0] } else { "$out\conteudo_doc\extraido_full.tsv" }
$loteDir="$out\lotes_ren"; New-Item -ItemType Directory -Path $loteDir -Force|Out-Null
Get-ChildItem $loteDir -File -EA SilentlyContinue|Remove-Item -Force
$outDir="$out\lotes_ren_out"; New-Item -ItemType Directory -Path $outDir -Force|Out-Null
Get-ChildItem $outDir -File -EA SilentlyContinue|Remove-Item -Force

$map=New-Object System.Collections.Generic.List[object]
$buf=New-Object System.Collections.Generic.List[string]
$id=0;$li=0;$size=200;$comTexto=0
foreach($line in [IO.File]::ReadLines($tsv,[Text.Encoding]::UTF8)){
  $p=$line -split "`t",3
  if($p.Count -lt 3){ continue }
  $path=$p[0]; $len=[int]$p[1]; $txt=$p[2]
  if($len -lt 60){ continue }
  $win = ($path -replace '^/c/','C:/' -replace '^/C/','C:/') -replace '/','\'
  if(-not (Test-Path -LiteralPath $win)){ continue }
  # pasta-contexto relativa (categoria\subpasta...)
  $rel = $win.Substring($base.Length+1)
  $pasta = [IO.Path]::GetDirectoryName($rel); if([string]::IsNullOrEmpty($pasta)){ $pasta='(raiz)' }
  $nome = [IO.Path]::GetFileName($win)
  $comTexto++; $id++
  $trecho = if($txt.Length -gt 500){ $txt.Substring(0,500) } else { $txt }
  $buf.Add(("{0}`t{1}`t{2}`t{3}" -f $id, $pasta, $nome, $trecho))
  $map.Add([pscustomobject]@{ ID=$id; Path=$win; Pasta=$pasta; Nome=$nome; Ext=[IO.Path]::GetExtension($win) })
  if($buf.Count -ge $size){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $buf.Clear(); $li++ }
}
if($buf.Count){ Set-Content -LiteralPath ("{0}\lote_{1:D3}.txt" -f $loteDir,$li) -Value $buf -Encoding UTF8; $li++ }
$map | Export-Csv "$out\map_ren.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Com texto p/ renomear: {0} | lotes: {1}" -f $comTexto, $li)