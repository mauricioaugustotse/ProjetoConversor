# Move soltos da raiz de Juridico/Estudos para a subpasta indicada pela IA. Pareia por ID.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cfg=@(@{Cat='05 - Jurídico e trabalho';Map='map_jur.csv';Res='res_jur.csv'},@{Cat='06 - Estudos, concursos e leitura';Map='map_est.csv';Res='res_est.csv'})
$fix=Get-Content -LiteralPath "$out\fix_dest.json" -Encoding UTF8 | ConvertFrom-Json
$plan=New-Object System.Collections.Generic.List[object]; $manter=0
foreach($c in $cfg){
  $root=Join-Path $base $c.Cat
  $map=@{}; foreach($m in (Import-Csv "$out\$($c.Map)" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
  $resPath="$out\lotes_recl_out\$($c.Res)"
  if(-not (Test-Path -LiteralPath $resPath)){ Write-Output ("FALTA res: {0}" -f $c.Res); continue }
  $cls=@{}; foreach($r in (Import-Csv -LiteralPath $resPath -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.destino } }
  foreach($id in ($map.Keys|Sort-Object)){
    $d=$cls[$id]
    if(-not $d -or $d -eq 'manter'){ $manter++; continue }
    if($fix.$d){ $d=$fix.$d }
    $destDir=Join-Path $root $d
    $plan.Add([pscustomobject]@{ De=$map[$id].Full; Para=(Join-Path $destDir $map[$id].Nome); Cat=$c.Cat; Dest=$d })
  }
}
Write-Output ("A mover: {0} | manter na raiz: {1}" -f $plan.Count,$manter)
$plan|Group-Object Dest|Sort-Object Count -Descending|Select-Object -First 25|ForEach-Object{ Write-Output ("   {0,3}  {1}" -f $_.Count,$_.Name) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_recl_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ $dir=[IO.Path]::GetDirectoryName($p.Para); [IO.Directory]::CreateDirectory($dir)|Out-Null
      $dest=$p.Para; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $dir ("$b ($i)$e"); $i++ } }
      [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++
    }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}