# Concentra todos os prompts em "12 - Consultoria Legislativa\PROMPTs", deduplica e padroniza nomes.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cat12=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '12 *' } | Select-Object -First 1
$dest=Get-ChildItem -LiteralPath $cat12.FullName -Directory -Force | Where-Object { $_.Name -like 'PROMPT*' } | Select-Object -First 1
$destDir=$dest.FullName
$rows=Import-Csv "$out\prompts.csv" -Encoding UTF8

function Padroniza($nome){
  $n=[IO.Path]::GetFileNameWithoutExtension($nome)
  $dash=[char]0x2013; $trim=[char[]]@('-',' ',$dash,'.')
  $n=$n -replace '(?i)^\s*prompts?\s+(para|de|da|do)\s+',''
  $n=$n -replace '(?i)^\s*prompts?\s*[-:]?\s*',''
  $n=$n -replace ('(?i)\s*[-(' + $dash + ']\s*prompts?\s*\)?\s*$'),''
  $n=($n -replace '\s{2,}',' ').Trim()
  $n=$n.Trim($trim)
  if($n.Length -lt 3){ $n=[IO.Path]::GetFileNameWithoutExtension($nome) }
  if($n){ $n=$n.Substring(0,1).ToUpper()+$n.Substring(1) }
  return $n
}
# dedup por hash: canonico = preferir o que ja esta em 12
$plan=New-Object System.Collections.Generic.List[object]; $usado=@{}
foreach($g in ($rows|Group-Object Hash)){
  $itens=@($g.Group)
  if($g.Name -eq 'ERR'){ foreach($it in $itens){ $plan.Add([pscustomobject]@{Acao='Mover';De=$it.Full;Nome=$it.Nome;Ext=$it.Ext}) }; continue }
  $canon=$itens|Where-Object{$_.Pasta -like '12 *'}|Select-Object -First 1; if(-not $canon){ $canon=$itens[0] }
  $plan.Add([pscustomobject]@{Acao='Mover';De=$canon.Full;Nome=$canon.Nome;Ext=$canon.Ext})
  foreach($it in $itens){ if($it.Full -ne $canon.Full){ $plan.Add([pscustomobject]@{Acao='Lixeira';De=$it.Full;Nome=$it.Nome;Ext=$it.Ext}) } }
}
# resolver destino dos Mover
$final=New-Object System.Collections.Generic.List[object]
foreach($p in ($plan|Where-Object Acao -eq 'Mover')){
  $novo=Padroniza $p.Nome; $name="$novo$($p.Ext)"; $d=Join-Path $destDir $name; $b=$novo; $i=2
  while($usado.ContainsKey($d.ToLower()) -or (Test-Path -LiteralPath $d)){ $name="$b ($i)$($p.Ext)"; $d=Join-Path $destDir $name; $i++ }
  $usado[$d.ToLower()]=1
  if($d -ne $p.De){ $final.Add([pscustomobject]@{De=$p.De;Para=$d}) }
}
$mv=$final; $lx=@($plan|Where-Object Acao -eq 'Lixeira')
Write-Output ("Mover p/ 12\PROMPTs: {0} | Duplicatas p/ Lixeira: {1}" -f $mv.Count,$lx.Count)
$mv|Select-Object -First 18|ForEach-Object{ Write-Output ("   {0}`n     -> {1}" -f (Split-Path $_.De -Leaf),(Split-Path $_.Para -Leaf)) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_prompts_$ts.csv"
  $r=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0;$okl=0
  foreach($p in $mv){ try{ [IO.File]::Move($p.De,$p.Para); $r.Add([pscustomobject]@{Acao='Mover';De=$p.De;Para=$p.Para}); $ok++ }catch{ $err++ } }
  foreach($p in $lx){ try{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p.De,'OnlyErrorDialogs','SendToRecycleBin'); $r.Add([pscustomobject]@{Acao='Lixeira';De=$p.De;Para=''}); $okl++ }catch{ $err++ } }
  $r|Export-Csv $log -NoTypeInformation -Encoding UTF8
  # remover pastas de origem que se chamam PROMPTS e ficaram vazias
  $vaz=Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -EA SilentlyContinue | Where-Object { $_.Name -match '(?i)^prompts' -and $_.FullName -ne $destDir -and @(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force -EA SilentlyContinue|Where-Object{$_.Name -ne 'desktop.ini'}).Count -eq 0 }
  $remP=0; foreach($v in $vaz){ try{[IO.Directory]::Delete($v.FullName,$true);$remP++}catch{} }
  Write-Output ("`nMovidos: {0} | Lixeira: {1} | Erros: {2} | Pastas PROMPTS vazias removidas: {3} | Log: {4}" -f $ok,$okl,$err,$remP,$log)
}