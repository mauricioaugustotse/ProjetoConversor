# Confirma duplicatas REAIS por hash MD5 (somente candidatos de mesmo tamanho). NAO apaga nada: gera plano em CSV.
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$idx = Import-Csv "$out\indice_mestre.csv" -Encoding UTF8
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$cand = $idx | Where-Object { [long]$_.Length -gt 0 } | Group-Object Length | Where-Object Count -gt 1
$rows = New-Object System.Collections.Generic.List[object]
foreach($g in $cand){
  foreach($f in $g.Group){
    $h = (Get-FileHash -LiteralPath $f.FullName -Algorithm MD5 -ErrorAction SilentlyContinue).Hash
    if($h){ $rows.Add([pscustomobject]@{ Hash=$h; Length=[long]$f.Length; Rel=$f.Rel; Name=$f.Name; Modified=$f.Modified; IsTSE=$f.IsTSE; FullName=$f.FullName }) }
  }
}
$sw.Stop()

$dups = $rows | Group-Object Hash | Where-Object Count -gt 1
$plan = New-Object System.Collections.Generic.List[object]
[long]$recuperavel = 0
foreach($d in $dups){
  # Manter: preferir nome SEM marcador de copia, depois nome mais curto, depois mais antigo
  $items = $d.Group | Sort-Object @{e={ if($_.Name -match '(\(\d+\))|(C[oó]pia)|(Copy)'){1}else{0} }}, @{e={$_.Name.Length}}, Modified
  $keep = $items[0]
  for($i=1; $i -lt $items.Count; $i++){
    $plan.Add([pscustomobject]@{ Hash=$d.Name; Length=$items[$i].Length; Remover=$items[$i].Rel; Manter=$keep.Rel; RemoverIsTSE=$items[$i].IsTSE; FullRemover=$items[$i].FullName })
    $recuperavel += [long]$items[$i].Length
  }
}
$plan | Export-Csv "$out\duplicatas_plano.csv" -NoTypeInformation -Encoding UTF8

Write-Output ("Candidatos hasheados : {0:N0}  em {1:N0}s" -f $rows.Count, $sw.Elapsed.TotalSeconds)
Write-Output ("Conjuntos de duplicatas REAIS: {0:N0}" -f $dups.Count)
Write-Output ("Arquivos duplicados a remover: {0:N0}" -f $plan.Count)
Write-Output ("Espaco recuperavel REAL      : {0:N2} GB" -f ($recuperavel/1GB))
Write-Output "Plano salvo em $out\duplicatas_plano.csv (NADA foi apagado)"