# Organiza o imovel Mykonos em subpastas tematicas (junta POUPEX). Pareia por ID global.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$imov=Join-Path $base (Get-ChildItem -LiteralPath $base -Directory|Where-Object{$_.Name -like '04 *'}|Select-Object -First 1).Name
$myk=Join-Path $imov 'MYKONOS'; $pou=Join-Path $imov 'POUPEX'
$sub=Get-Content -LiteralPath "$out\sub_mykonos.json" -Encoding UTF8 | ConvertFrom-Json
$map=@{}; foreach($m in (Import-Csv "$out\map_myk.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}
foreach($f in (Get-ChildItem "$out\lotes_myk_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.categoria } } }
Write-Output ("Map: {0} | classificados: {1}" -f $map.Count,$cls.Count)

$plan=New-Object System.Collections.Generic.List[object]; $usado=@{}
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $key=$cls[$id]; if(-not $sub.$key){ $key='Outros' }
  $destDir=Join-Path $myk $sub.$key
  $nome=$map[$id].Nome; $dest=Join-Path $destDir $nome
  $b=[IO.Path]::GetFileNameWithoutExtension($nome); $e=[IO.Path]::GetExtension($nome); $i=2
  while($usado.ContainsKey($dest.ToLower()) -or (Test-Path -LiteralPath $dest)){ $dest=Join-Path $destDir ("$b ($i)$e"); $i++ }
  if($dest -eq $map[$id].Full){ continue }
  $usado[$dest.ToLower()]=1
  $plan.Add([pscustomobject]@{ De=$map[$id].Full; Para=$dest; Cat=$sub.$key })
}
Write-Output ("A mover: {0}" -f $plan.Count)
$plan | Group-Object Cat | Sort-Object Count -Descending | ForEach-Object { Write-Output ("   {0,-34} {1}" -f $_.Name,$_.Count) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_myk_$ts.csv"
  foreach($k in $sub.PSObject.Properties.Name){ [IO.Directory]::CreateDirectory((Join-Path $myk $sub.$k))|Out-Null }
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){ try{ [IO.File]::Move($p.De,$p.Para); $rows.Add([pscustomobject]@{De=$p.De;Para=$p.Para}); $ok++ }catch{ $err++ } }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  # remover POUPEX se vazio
  $cascaMsg='POUPEX mantido'
  if(Test-Path -LiteralPath $pou){ $rest=@(Get-ChildItem -LiteralPath $pou -Recurse -File -Force -EA SilentlyContinue|Where-Object{$_.Name -ne 'desktop.ini'}).Count; if($rest -eq 0){ try{[IO.Directory]::Delete($pou,$true); $cascaMsg='pasta POUPEX removida (vazia)'}catch{$cascaMsg="POUPEX nao removido: $($_.Exception.Message)"} } else { $cascaMsg="POUPEX ainda com $rest arquivos" } }
  Write-Output ("`nMovidos: {0} | Erros: {1} | {2} | Log: {3}" -f $ok,$err,$cascaMsg,$log)
}