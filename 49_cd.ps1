# Extrai posse + processos do concurso CD para 01\<dedicada>. Material de estudo fica em 06.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cd="$base\06 - Estudos, concursos e leitura\1_CONCURSOS\CD"
$cat01=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '01 *' } | Select-Object -First 1
$cfg=Get-Content -LiteralPath "$out\config_cd.json" -Encoding UTF8 | ConvertFrom-Json
$destRoot=Join-Path $cat01.FullName $cfg.destino
$map=@{}; foreach($m in (Import-Csv "$out\map_cd.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}
foreach($f in (Get-ChildItem "$out\lotes_cd_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.classe } } }

$plan=New-Object System.Collections.Generic.List[object]
# subpastas inteiras
foreach($s in $cfg.subpastas){
  $src=Join-Path $cd $s
  if(Test-Path -LiteralPath $src){ $plan.Add([pscustomobject]@{Tipo='Dir';De=$src;Para=(Join-Path $destRoot $s)}) }
}
# soltos POSSE/PROCESSO
$nP=0
foreach($id in ($map.Keys|Sort-Object)){
  if($cls[$id] -eq 'POSSE' -or $cls[$id] -eq 'PROCESSO'){ $plan.Add([pscustomobject]@{Tipo='File';De=$map[$id].Full;Para=(Join-Path $destRoot $map[$id].Nome)}); $nP++ }
}
Write-Output ("Destino: 01\{0}" -f $cfg.destino)
Write-Output ("Subpastas a mover: {0} | Soltos (posse/processo) a mover: {1}" -f @($plan|Where-Object Tipo -eq 'Dir').Count, $nP)
Write-Output ("Ficam em 06\CD (estudo): {0} soltos + subpastas de estudo" -f @($cls.Values|Where-Object {$_ -eq 'ESTUDO'}).Count)
$plan|Where-Object Tipo -eq 'Dir'|ForEach-Object{ Write-Output ("   [pasta] {0}" -f (Split-Path $_.De -Leaf)) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_cd_$ts.csv"
  [IO.Directory]::CreateDirectory($destRoot)|Out-Null
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{
      $dest=$p.Para
      if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $d=[IO.Path]::GetDirectoryName($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $d ("$b ($i)$e"); $i++ } }
      if($p.Tipo -eq 'Dir'){ [IO.Directory]::Move($p.De,$dest) } else { [IO.File]::Move($p.De,$dest) }
      $rows.Add([pscustomobject]@{Tipo=$p.Tipo;De=$p.De;Para=$dest}); $ok++
    }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}