# Move docs pessoais profundos: Mauricio -> 01\<tipo>, Familia -> 09, Fica -> nao move. Pareia por ID.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cats=@{}; foreach($d in (Get-ChildItem -LiteralPath $base -Directory -Force)){ if($d.Name -match '^(\d{2}) '){ $cats[$Matches[1]]=$d.FullName } }
$destChave=Get-Content -LiteralPath "$out\destinos_cofre.json" -Encoding UTF8|ConvertFrom-Json
$map=@{}; foreach($m in (Import-Csv "$out\map_prof.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($f in (Get-ChildItem "$out\lotes_prof_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.chave } } }
function DestDir($spec){ $p=$spec -split '\|',2; $dir=$cats[$p[0]]; if($p.Count -gt 1 -and $p[1]){ $dir=Join-Path $dir $p[1] }; return $dir }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  $ch=$cls[$id]; if(-not $ch -or $ch -eq 'Fica'){ continue }
  if(-not $destChave.$ch){ continue }
  # Familia: nao mexer em processos (05), dossie de imovel (04), ja-familia (09) ou senhas (10)
  $ctxCat=$map[$id].Full.Substring($base.Length+1).Split('\')[0]
  if($ch -eq 'Familia' -and ($ctxCat -like '04*' -or $ctxCat -like '05*' -or $ctxCat -like '09*' -or $ctxCat -like '10*')){ continue }
  $plan+=[pscustomobject]@{ De=$map[$id].Full; Para=(Join-Path (DestDir $destChave.$ch) $map[$id].Nome); Chave=$ch }
}
Write-Output ("A mover: {0} | Ficam (fiscal/contexto): {1}" -f $plan.Count, @($cls.Values|Where-Object{$_ -eq 'Fica'}).Count)
$plan|Group-Object{ $_.Para.Substring($base.Length+1).Split('\')[0..1] -join '\' }|Sort-Object Count -Descending|ForEach-Object{ Write-Output ("   {0,3}  {1}" -f $_.Count,$_.Name) }
Write-Output "`n-- amostra --"
$plan|Select-Object -First 12|ForEach-Object{ Write-Output ("   [{0}] {1}" -f $_.Chave,(Split-Path $_.De -Leaf)) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_prof_$ts.csv"
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