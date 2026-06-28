# Fase 4b: sub-organiza PROMPTs por finalidade. Pareia por ID. Sem hardcode de acentos.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$c12=Get-ChildItem -LiteralPath $base -Directory -Force|Where-Object{$_.Name -like '12 *'}|Select-Object -First 1
$pr=Get-ChildItem -LiteralPath $c12.FullName -Directory -Force|Where-Object{$_.Name -like 'PROMPT*'}|Select-Object -First 1
$prPath=$pr.FullName
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$map=@{}; foreach($m in (Import-Csv "$out\map_prfin.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($r in (Import-Csv "$out\lotes_prfin_out\res_00.csv" -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.subpasta } }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $sub=Sanitize $cls[$id]; if(-not $sub){ continue }
  $plan+=[pscustomobject]@{ De=$map[$id].Full; DestDir=(Join-Path $prPath $sub); Nome=$map[$id].Nome; Sub=$sub }
}
Write-Output ("PROMPTs: {0} | subpastas: {1}" -f $plan.Count, @($plan|Group-Object Sub).Count)
$plan|Group-Object Sub|Sort-Object Count -Descending|ForEach-Object{ Write-Output ("   {0,2}  {1}" -f $_.Count,$_.Name) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_prfin_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){ try{ [IO.Directory]::CreateDirectory($p.DestDir)|Out-Null; $dest=Join-Path $p.DestDir $p.Nome; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $p.DestDir ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ } }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}