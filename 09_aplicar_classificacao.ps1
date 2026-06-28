# Regera o plano final combinando regras + classificacao da IA (override nos Diversos). Dry-run: NAO move/apaga.
$ErrorActionPreference = 'SilentlyContinue'
$out  = 'C:\Users\mauri\ProjetoConversor'
$base = 'C:\Users\mauri\HD_Mau'
$idx  = Import-Csv "$out\indice_mestre.csv" -Encoding UTF8
$cfg  = Get-Content -Raw -Encoding UTF8 "$out\regras.json" | ConvertFrom-Json

$lixoExt  = $cfg.lixo.extensoes
$lixoNome = ($cfg.lixo.nomePatterns -join '|')
$rules    = $cfg.regras
$reFim    = $cfg.normalizacao.removerFim
$reIni    = $cfg.normalizacao.removerInicio
$natsOk   = $cfg.taxonomia.nats
$tiposJur = $cfg.taxonomia.tiposJuridico
$jurNat   = $cfg.taxonomia.juridicoNat
$jurDef   = $cfg.taxonomia.tipoJuridicoDefault

function NewBase($name){
  $ext = [IO.Path]::GetExtension($name).ToLower()
  $b   = [IO.Path]::GetFileNameWithoutExtension($name)
  foreach($p in $reFim){ $b = $b -replace $p, '' }
  foreach($p in $reIni){ $b = $b -replace $p, '' }
  $b = ($b -replace '\s{2,}', ' ').Trim()
  if([string]::IsNullOrWhiteSpace($b)){ $b = [IO.Path]::GetFileNameWithoutExtension($name) }
  return ,@($b, $ext)
}

# ---- Carrega override da IA: Origem(FullName) -> @(nat,tipo) ----
$resById = @{}
foreach($file in (Get-ChildItem "$out\lotes_out\res_*.csv")){
  foreach($row in (Import-Csv -LiteralPath $file.FullName -Encoding UTF8)){
    if($row.id){ $resById[[int]$row.id] = @([string]$row.nat, [string]$row.tipo) }
  }
}
$ov = @{}
foreach($m in (Import-Csv "$out\diversos_map.csv" -Encoding UTF8)){
  $id = [int]$m.ID
  if($resById.ContainsKey($id)){ $ov[$m.Origem] = $resById[$id] }
}
Write-Output ("Override IA: {0:N0} arquivos com classificacao (de {1:N0} esperados)" -f $ov.Count, (Import-Csv "$out\diversos_map.csv" -Encoding UTF8).Count)

$plan = New-Object System.Collections.Generic.List[object]
$used = @{}
$semOverride = 0
foreach($f in $idx){
  $name = $f.Name; $ext = ([string]$f.Ext).ToLower()

  $isLixo = $false
  if($lixoExt -contains $ext){ $isLixo = $true }
  elseif(($ext -eq '.html' -or $ext -eq '.htm') -and $name -match "(?i)($lixoNome)"){ $isLixo = $true }
  elseif($name -match '(?i)saved_resource'){ $isLixo = $true }
  if($isLixo){
    $plan.Add([pscustomobject]@{ Acao='APAGAR'; Origem=$f.FullName; Destino=''; Nat='99 - Lixo web'; Tipo=''; NomeNovo='' })
    continue
  }

  $nb = NewBase $name; $b = $nb[0]; $e = $nb[1]
  $mk = $b -replace '^\s*[\(\[]\d+[\)\]]\s*', ''

  # 1) regras determinADASisticas
  $nat = $cfg.default.nat; $tipo = $cfg.default.tipo
  foreach($r in $rules){ if($mk -match ('(?i)' + $r.pat)){ $nat = $r.nat; $tipo = $r.tipo; break } }

  # 2) se caiu em Diversos, usa override da IA (quando houver)
  if($nat -like '05*'){
    if($ov.ContainsKey($f.FullName)){ $nat = $ov[$f.FullName][0]; $tipo = $ov[$f.FullName][1] }
    else { $semOverride++ }
  }

  # 3) validacao da taxonomia
  if($natsOk -notcontains $nat){ $nat = $cfg.default.nat; $tipo = '' }
  if($nat -eq $jurNat){ if($tiposJur -notcontains $tipo){ $tipo = $jurDef } } else { $tipo = '' }

  if($tipo){ $destDir = Join-Path $nat $tipo } else { $destDir = $nat }
  $newName = $b + $e
  $relDest = Join-Path $destDir $newName
  $key = $relDest.ToLower()
  if($used.ContainsKey($key)){
    $i = 2
    do { $nn = "$b ($i)$e"; $rd = Join-Path $destDir $nn; $k2 = $rd.ToLower(); $i++ } while($used.ContainsKey($k2))
    $relDest = $rd; $newName = $nn; $key = $k2
  }
  $used[$key] = 1
  $plan.Add([pscustomobject]@{ Acao='MOVER'; Origem=$f.FullName; Destino=(Join-Path $base $relDest); Nat=$nat; Tipo=$tipo; NomeNovo=$newName })
}
$plan | Export-Csv "$out\plano_organizacao.csv" -NoTypeInformation -Encoding UTF8

Write-Output ("`nArquivos em Diversos sem override IA: {0:N0}" -f $semOverride)
Write-Output ("Total no plano : {0:N0}" -f $plan.Count)
Write-Output ("APAGAR (lixo)  : {0:N0}" -f (($plan | Where-Object Acao -eq 'APAGAR') | Measure-Object).Count)
Write-Output ("MOVER          : {0:N0}" -f (($plan | Where-Object Acao -eq 'MOVER')  | Measure-Object).Count)
Write-Output "`n=== Distribuicao final por categoria ==="
$plan | Where-Object Acao -eq 'MOVER' | Group-Object Nat, Tipo | Sort-Object Count -Descending |
  ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Count, $_.Name) }