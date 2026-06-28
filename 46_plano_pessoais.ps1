# Centraliza documentos pessoais do Maurico em 01\<Tipo>, removendo duplicatas exatas (mesmo hash).
# Tipos (com acento) lidos de tipos_pess.json (UTF-8). Pareia classificacao por ID global.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cat01=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '01 *' } | Select-Object -First 1
$root01=$cat01.FullName
$tipos=Get-Content -LiteralPath "$out\tipos_pess.json" -Encoding UTF8 | ConvertFrom-Json
$map=@{}; foreach($m in (Import-Csv "$out\map_pess.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}
foreach($f in (Get-ChildItem "$out\lotes_pess_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r } } }

# itens MAURICIO com tipo valido
$maur=@()
foreach($id in $map.Keys){ if($cls.ContainsKey($id) -and $cls[$id].titular -eq 'MAURICIO'){ $t=$cls[$id].tipo; if($t -and $tipos.$t){ $maur+=[pscustomobject]@{ ID=$id; Full=$map[$id].Full; Hash=$map[$id].Hash; Nome=$map[$id].Nome; Tipo=$t; EmPasta=$map[$id].Pasta } } } }
Write-Output ("Documentos MAURICIO a centralizar: {0}" -f $maur.Count)

$plan=New-Object System.Collections.Generic.List[object]
$usado=@{}
function DestinoUnico($dir,$nome){
  $dest=Join-Path $dir $nome; $b=[IO.Path]::GetFileNameWithoutExtension($nome); $e=[IO.Path]::GetExtension($nome); $i=2
  while($script:usado.ContainsKey($dest.ToLower()) -or (Test-Path -LiteralPath $dest)){ $dest=Join-Path $dir ("$b ($i)$e"); $i++ }
  $script:usado[$dest.ToLower()]=1; return $dest
}
foreach($g in ($maur | Group-Object Hash)){
  $itens=@($g.Group)
  $tipo=($itens | Where-Object {$_.Tipo} | Select-Object -First 1).Tipo
  $pastaTipo=Join-Path $root01 $tipos.$tipo
  # canonico: preferir o que ja esta em 01
  $canon=$itens | Where-Object { $_.Full -like "$root01*" } | Select-Object -First 1
  if(-not $canon){ $canon=$itens[0] }
  $jaNoDestino = ([IO.Path]::GetDirectoryName($canon.Full) -eq $pastaTipo)
  if($jaNoDestino){ $destC=$canon.Full } else { $destC=DestinoUnico $pastaTipo $canon.Nome }
  if($destC -ne $canon.Full){ $plan.Add([pscustomobject]@{ Acao='Mover'; De=$canon.Full; Para=$destC; Tipo=$tipo }) }
  foreach($o in $itens){ if($o.Full -ne $canon.Full){ $plan.Add([pscustomobject]@{ Acao='Lixeira'; De=$o.Full; Para=''; Tipo=$tipo }) } }
}
$plan | Export-Csv "$out\plano_pess.csv" -NoTypeInformation -Encoding UTF8
$mv=@($plan|Where-Object Acao -eq 'Mover'); $lx=@($plan|Where-Object Acao -eq 'Lixeira')
Write-Output ("Mover p/ 01\<tipo>: {0} | Duplicatas p/ Lixeira: {1}" -f $mv.Count,$lx.Count)
Write-Output "`n-- por tipo (mover) --"
$mv | Group-Object Tipo | Sort-Object Count -Descending | ForEach-Object { Write-Output ("   {0,-12} {1}" -f $_.Name,$_.Count) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_pess_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $okM=0;$okL=0;$err=0
  foreach($t in $tipos.PSObject.Properties.Name){ $d=Join-Path $root01 $tipos.$t; [IO.Directory]::CreateDirectory($d)|Out-Null }
  foreach($p in ($plan|Where-Object Acao -eq 'Mover')){ try{ [IO.File]::Move($p.De,$p.Para); $rows.Add([pscustomobject]@{Acao='Mover';De=$p.De;Para=$p.Para}); $okM++ }catch{ $err++ } }
  foreach($p in ($plan|Where-Object Acao -eq 'Lixeira')){ try{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p.De,'OnlyErrorDialogs','SendToRecycleBin'); $rows.Add([pscustomobject]@{Acao='Lixeira';De=$p.De;Para=''}); $okL++ }catch{ $err++ } }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Lixeira: {1} | Erros: {2} | Log: {3}" -f $okM,$okL,$err,$log)
}