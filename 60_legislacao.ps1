# Fase 2: renomeia pares csv+md de Legislação Total para nomes legiveis. Pareia por ID.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$out='C:\Users\mauri\ProjetoConversor'
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$map=@{}; foreach($m in (Import-Csv "$out\map_leg.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($r in (Import-Csv "$out\lotes_leg_out\res_00.csv" -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.nomeNovo } }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $novo=Sanitize $cls[$id]; if(-not $novo){ continue }
  foreach($a in ($map[$id].Arquivos -split '\|')){
    $ext=[IO.Path]::GetExtension($a); $dir=[IO.Path]::GetDirectoryName($a)
    $plan+=[pscustomobject]@{ De=$a; Para=(Join-Path $dir ($novo+$ext)); Base=$map[$id].Base }
  }
}
Write-Output ("Arquivos a renomear: {0} (de {1} leis)" -f $plan.Count, @($plan|Group-Object Base).Count)
$plan|Where-Object{$_.De -like '*.csv'}|ForEach-Object{ Write-Output ("   {0}  ->  {1}" -f (Split-Path $_.De -Leaf),(Split-Path $_.Para -Leaf)) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_leg_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ $dest=$p.Para; if((Test-Path -LiteralPath $dest) -and ($dest -ne $p.De)){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $d=[IO.Path]::GetDirectoryName($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $d ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}