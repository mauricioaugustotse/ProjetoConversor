# Monta plano de reorganizacao dos Diversos a partir da classificacao por conteudo (Haiku). Dry-run.
$ErrorActionPreference = 'SilentlyContinue'
$out  = 'C:\Users\mauri\ProjetoConversor'
$base = 'C:\Users\mauri\HD_Mau'

$nats = @('01 - Juridico','02 - Estudos e concursos','03 - Administrativo','04 - Pessoal e financeiro','05 - Diversos (a revisar)')
$tiposJur = @('Recursos','Peticoes','Acordaos e decisoes','Despachos e votos','Contestacoes e manifestacoes','Pareceres e notas tecnicas','Modelos e minutas','Outros processuais')

# map ID -> arquivo
$map = @{}
foreach($m in (Import-Csv "$out\map_div.csv" -Encoding UTF8)){ $map[[int]$m.ID] = $m }

# classificacoes pareadas por POSICAO (lote <-> res)
$cls = @{}
for($li=0; $li -le 60; $li++){
  $loteFile = "{0}\lotes_div\lote_{1:D3}.txt" -f $out,$li
  $resFile  = "{0}\lotes_div_out\res_{1:D3}.csv" -f $out,$li
  if(-not (Test-Path -LiteralPath $resFile)){ continue }
  $loteLines = @(Get-Content -LiteralPath $loteFile -Encoding UTF8 | Where-Object { $_.Trim() -ne '' })
  $resRows   = @(Import-Csv -LiteralPath $resFile -Delimiter '|' -Encoding UTF8)
  $cnt = [Math]::Min($loteLines.Count, $resRows.Count)
  for($k=0; $k -lt $cnt; $k++){
    $gid = [int](($loteLines[$k] -split "`t",2)[0])
    $cls[$gid] = $resRows[$k]
  }
}
Write-Output ("Classificacoes pareadas: {0}" -f $cls.Count)

function SanitizeName($s){
  if([string]::IsNullOrWhiteSpace($s)){ return '' }
  $s = $s -replace '[\\/:\*\?"<>\|]', ' '
  $s = ($s -replace '\s{2,}',' ').Trim().TrimEnd('.',' ')
  return $s
}

$plan = New-Object System.Collections.Generic.List[object]
$used = @{}
$movem=0; $ficam=0
foreach($id in ($map.Keys | Sort-Object)){
  $a = $map[$id]
  if(-not $cls.ContainsKey($id)){ continue }
  $c = $cls[$id]
  $nat=$c.nat; $tipo=$c.tipo; $mat=$c.materia; $nome=$c.nomeNovo
  if($nats -notcontains $nat){ $nat='05 - Diversos (a revisar)' }
  # os que permanecem em Diversos ficam INTACTOS (nome e local originais) - so agimos nos que saem p/ categorias reais
  if($nat -like '05*'){ $ficam++; continue }
  if($nat -eq '01 - Juridico'){ if($tiposJur -notcontains $tipo){ $tipo='Outros processuais' } } else { $tipo='' }
  if([string]::IsNullOrWhiteSpace($mat)){ $mat='Outros' }
  $nome = SanitizeName $nome
  if([string]::IsNullOrWhiteSpace($nome)){ $nome = [IO.Path]::GetFileNameWithoutExtension($a.NomeAtual) }
  $ext = $a.Ext

  if($nat -eq '01 - Juridico'){ $destDir = Join-Path (Join-Path (Join-Path $base $nat) $tipo) $mat }
  else { $destDir = Join-Path (Join-Path $base $nat) $mat }

  $reserva = $destDir.Length + 1 + $ext.Length + 6
  $maxNome = [Math]::Max(20, 255 - $reserva)
  if($nome.Length -gt $maxNome){ $nome = $nome.Substring(0,$maxNome).TrimEnd('.',' ') }

  $newName = $nome + $ext
  $relDest = Join-Path $destDir $newName
  $key = $relDest.ToLower()
  while($used.ContainsKey($key) -or (Test-Path -LiteralPath $relDest)){
    $i = if($used.ContainsKey($key+'#c')){ $used[$key+'#c']+1 } else { 2 }
    $used[$key+'#c'] = $i
    $newName = "$nome ($i)$ext"; $relDest = Join-Path $destDir $newName; $key = $relDest.ToLower()
    if($i -gt 50){ break }
  }
  $used[$key] = 1
  if($a.Path -eq $relDest){ $ficam++; continue }
  $plan.Add([pscustomobject]@{ Origem=$a.Path; Destino=$relDest; Nat=$nat; Tipo=$tipo; Materia=$mat; NomeNovo=$newName })
  $movem++
}
$plan | Export-Csv "$out\plano_div.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("No plano: {0} | ja no lugar: {1}" -f $movem, $ficam)
Write-Output "`n=== Para onde vao (Diversos -> destino) ==="
$plan | Group-Object Nat | Sort-Object Count -Descending | ForEach-Object { Write-Output ("  {0,5}  {1}" -f $_.Count, $_.Name) }
$saiem = @($plan | Where-Object { $_.Nat -notlike '05*' }).Count
Write-Output ("`nSaem de Diversos para categorias reais: {0} de {1}" -f $saiem, $plan.Count)