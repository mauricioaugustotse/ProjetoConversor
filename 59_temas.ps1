# Fase 1: agrupa "Temas Selecionados" em subpasta por tema. Pareia por ID. Sem hardcode de acentos.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$c12=Get-ChildItem -LiteralPath $base -Directory -Force|Where-Object{$_.Name -like '12 *'}|Select-Object -First 1
$ts=Get-ChildItem -LiteralPath $c12.FullName -Directory -Force|Where-Object{$_.Name -like 'Temas*'}|Select-Object -First 1
$tsPath=$ts.FullName
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$map=@{}; foreach($m in (Import-Csv "$out\map_temas.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($r in (Import-Csv "$out\lotes_temas_out\res_00.csv" -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.subpasta } }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $sub=Sanitize $cls[$id]; if(-not $sub){ continue }
  $plan+=[pscustomobject]@{ De=$map[$id].Full; DestDir=(Join-Path $tsPath $sub); Nome=$map[$id].Nome; Tema=$sub }
}
$temas=@($plan|Group-Object Tema)
Write-Output ("Arquivos: {0} | temas (subpastas): {1}" -f $plan.Count,$temas.Count)
$temas|Sort-Object Name|ForEach-Object{ Write-Output ("   {0,2} arq  {1}" -f $_.Count,$_.Name) }
if($Execute){
  $tsLog=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_temas_$tsLog.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ [IO.Directory]::CreateDirectory($p.DestDir)|Out-Null; $dest=Join-Path $p.DestDir $p.Nome; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $p.DestDir ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}