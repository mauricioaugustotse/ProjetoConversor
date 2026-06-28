# Fase B: achata as pastas marcadas ACHATAR (sobe conteudo p/ a mae, remove a pasta). Pareia por ID global.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$out='C:\Users\mauri\ProjetoConversor'
$map=@{}; foreach($m in (Import-Csv "$out\map_flat.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$ach=@{}
foreach($f in (Get-ChildItem "$out\lotes_flat_out\res_*.csv")){
  foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){
    if($r.id -match '^\d+$' -and $r.decisao -match 'ACHATAR'){ $ach[[int]$r.id]=$true }
  }
}
# ordenar por profundidade desc (mais fundo primeiro) p/ evitar conflito de cascata
$alvos = $ach.Keys | Where-Object { $map.ContainsKey($_) } | Sort-Object { ($map[$_].Full -split '\\').Count } -Descending
Write-Output ("Pastas a achatar: {0}" -f @($alvos).Count)
$plan=New-Object System.Collections.Generic.List[object]
$raiz='C:\Users\mauri\OneDrive\Documentos'
foreach($id in $alvos){
  $full=$map[$id].Full; $mae=$map[$id].Mae
  if($mae -eq $raiz){ Write-Output ("  (pulando, mae=raiz): {0}" -f (Split-Path $full -Leaf)); continue }
  if(-not (Test-Path -LiteralPath $full)){ continue }
  $itens=@(Get-ChildItem -LiteralPath $full -Force -EA SilentlyContinue | Where-Object { $_.Name -ne 'desktop.ini' })
  foreach($it in $itens){ $tp=if($it.PSIsContainer){'D'}else{'F'}; $plan.Add([pscustomobject]@{ De=$it.FullName; Mae=$mae; Tipo=$tp; Pasta=$full }) }
}
Write-Output ("Itens a subir p/ a mae: {0}" -f $plan.Count)
$plan | Group-Object { Split-Path $_.Pasta -Leaf } | ForEach-Object { Write-Output ("   {0} itens <- {1}" -f $_.Count, $_.Name) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_achatar_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $okM=0;$err=0
  foreach($p in $plan){
    try{
      $nome=Split-Path $p.De -Leaf; $dest=Join-Path $p.Mae $nome
      if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($nome); $e=[IO.Path]::GetExtension($nome); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $p.Mae ("$b ($i)$e"); $i++ } }
      if($p.Tipo -eq 'D'){ [IO.Directory]::Move($p.De,$dest) } else { [IO.File]::Move($p.De,$dest) }
      $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $okM++
    }catch{ $err++ }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  # remover as pastas agora vazias
  $remP=0
  foreach($id in $alvos){ $full=$map[$id].Full; if(Test-Path -LiteralPath $full){ $rest=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force -EA SilentlyContinue | Where-Object {$_.Name -ne 'desktop.ini'}).Count; if($rest -eq 0){ try{ [IO.Directory]::Delete($full,$true); $remP++ }catch{} } } }
  Write-Output ("`nItens subidos: {0} | Erros: {1} | Pastas removidas: {2} | Log: {3}" -f $okM,$err,$remP,$log)
}