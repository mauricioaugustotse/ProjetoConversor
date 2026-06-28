# Classifica e gera plano de organizacao (dry-run; NAO move/apaga). Le regras.json (UTF-8) para evitar problemas de encoding.
$ErrorActionPreference = 'SilentlyContinue'
$out  = 'C:\Users\mauri\ProjetoConversor'
$base = 'C:\Users\mauri\HD_Mau'
$idx  = Import-Csv "$out\indice_mestre.csv" -Encoding UTF8
$cfg  = Get-Content -Raw -Encoding UTF8 "$out\regras.json" | ConvertFrom-Json

$lixoExt   = $cfg.lixo.extensoes
$lixoNome  = ($cfg.lixo.nomePatterns -join '|')
$rules     = $cfg.regras
$reFim     = $cfg.normalizacao.removerFim
$reIni     = $cfg.normalizacao.removerInicio

function NewBase($name){
  $ext = [IO.Path]::GetExtension($name).ToLower()
  $b   = [IO.Path]::GetFileNameWithoutExtension($name)
  foreach($p in $reFim){ $b = $b -replace $p, '' }
  foreach($p in $reIni){ $b = $b -replace $p, '' }
  $b = ($b -replace '\s{2,}', ' ').Trim()
  if([string]::IsNullOrWhiteSpace($b)){ $b = [IO.Path]::GetFileNameWithoutExtension($name) }
  return ,@($b, $ext)
}

$plan = New-Object System.Collections.Generic.List[object]
$used = @{}
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

  $nb = NewBase $name
  $b = $nb[0]; $e = $nb[1]
  # Chave de classificacao: ignora prefixo "(N)"/"[N]" numerico inicial (numeracao de juntada), mantido no nome final
  $mk = $b -replace '^\s*[\(\[]\d+[\)\]]\s*', ''
  $nat = $cfg.default.nat; $tipo = $cfg.default.tipo
  foreach($r in $rules){
    if($mk -match ('(?i)' + $r.pat)){ $nat = $r.nat; $tipo = $r.tipo; break }
  }
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

Write-Output ("Total no plano : {0:N0}" -f $plan.Count)
Write-Output ("APAGAR (lixo)  : {0:N0}" -f (($plan | Where-Object Acao -eq 'APAGAR') | Measure-Object).Count)
Write-Output ("MOVER          : {0:N0}" -f (($plan | Where-Object Acao -eq 'MOVER')  | Measure-Object).Count)
Write-Output "`n=== Distribuicao por categoria (MOVER) ==="
$plan | Where-Object Acao -eq 'MOVER' | Group-Object Nat, Tipo | Sort-Object Count -Descending |
  ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Count, $_.Name) }

$div = $plan | Where-Object { $_.Nat -like '05*' }
Write-Output ("`n=== Amostra de 'Diversos' ({0:N0} arquivos) ===" -f ($div | Measure-Object).Count)
$div | Select-Object -First 30 | ForEach-Object { Write-Output ("   {0}" -f (Split-Path $_.Origem -Leaf)) }