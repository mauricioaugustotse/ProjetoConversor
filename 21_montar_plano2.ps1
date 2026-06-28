# Monta o plano final de reorganizacao (subpastas por materia + nome novo). Dry-run: gera plano2.csv, nao move.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$tax  = Get-Content -Raw -Encoding UTF8 "$out\taxonomia2.json" | ConvertFrom-Json
$nats = $tax.nats; $tiposJur = $tax.tiposJur; $subs = $tax.subgrupos
$jurNat = $tax.juridicoNat; $divNat = $tax.diversosNat; $tipoDef = $tax.tipoJurDefault; $subDef = $tax.subgrupoDefault

# 1) Mapa ID -> arquivo
$map = @{}
foreach($m in (Import-Csv "$out\map2.csv" -Encoding UTF8)){ $map[[int]$m.ID] = $m }

# 2) Carregar classificacoes da IA por POSICAO (o id escrito pelos agentes e pouco confiavel;
#    pareia a k-esima linha do res com a k-esima linha do lote, que tem o ID global correto).
$cls = @{}
for($li=0; $li -le 144; $li++){
  $loteFile = "{0}\lotes2\lote_{1:D3}.txt" -f $out, $li
  $resFile  = "{0}\lotes2_out\res_{1:D3}.csv" -f $out, $li
  if(-not (Test-Path -LiteralPath $resFile)){ continue }
  $loteLines = @(Get-Content -LiteralPath $loteFile -Encoding UTF8 | Where-Object { $_.Trim() -ne '' })
  $resRows   = @(Import-Csv -LiteralPath $resFile -Delimiter '|' -Encoding UTF8)
  $cnt = [Math]::Min($loteLines.Count, $resRows.Count)
  for($k=0; $k -lt $cnt; $k++){
    $gid = [int](($loteLines[$k] -split "`t", 2)[0])
    $cls[$gid] = $resRows[$k]
  }
}
Write-Output ("Classificacoes pareadas por posicao: {0:N0}" -f $cls.Count)

function SanitizeName($s){
  if([string]::IsNullOrWhiteSpace($s)){ return '' }
  $s = $s -replace '[\\/:\*\?"<>\|]', ' '
  $s = $s -replace '[\x00-\x1F]', ' '
  $s = ($s -replace '\s{2,}', ' ').Trim()
  $s = $s.TrimEnd('.',' ')
  return $s
}

$plan = New-Object System.Collections.Generic.List[object]
$used = @{}
$semCls = 0
foreach($id in ($map.Keys | Sort-Object)){
  $f = $map[$id]
  $ext = $f.Ext; if([string]::IsNullOrWhiteSpace($ext)){ $ext = [IO.Path]::GetExtension($f.Name).ToLower() }
  $origNoExt = [IO.Path]::GetFileNameWithoutExtension($f.Name)

  $nat=$null;$tipo='';$mat='';$nome=''
  if($cls.ContainsKey($id)){
    $c = $cls[$id]; $nat=$c.nat; $tipo=$c.tipo; $mat=$c.materia; $nome=$c.nomeNovo
  } else { $semCls++; $nat=$f.Nat }

  # validar taxonomia
  if($nats -notcontains $nat){ $nat = $divNat }
  if($nat -eq $jurNat){ if($tiposJur -notcontains $tipo){ $tipo = $tipoDef } } else { $tipo = '' }
  if($subs -notcontains $mat){ $mat = $subDef }

  # nome novo
  $nome = SanitizeName $nome
  if([string]::IsNullOrWhiteSpace($nome)){ $nome = SanitizeName $origNoExt }
  if([string]::IsNullOrWhiteSpace($nome)){ $nome = "documento" }

  # diretorio destino
  if($nat -eq $jurNat){ $destDir = Join-Path (Join-Path (Join-Path $base $nat) $tipo) $mat }
  else { $destDir = Join-Path (Join-Path $base $nat) $mat }

  # limite de comprimento de caminho (<=255)
  $reserva = $destDir.Length + 1 + $ext.Length + 6   # 6 = margem p/ sufixo " (NN)"
  $maxNome = 255 - $reserva
  if($maxNome -lt 20){ $maxNome = 20 }
  if($nome.Length -gt $maxNome){ $nome = $nome.Substring(0, $maxNome).TrimEnd('.',' ') }

  $newName = $nome + $ext
  $relDest = Join-Path $destDir $newName
  $key = $relDest.ToLower()
  if($used.ContainsKey($key)){
    $i=2
    do { $nn = "$nome ($i)$ext"; $rd = Join-Path $destDir $nn; $k2=$rd.ToLower(); $i++ } while($used.ContainsKey($k2))
    $relDest=$rd; $newName=$nn; $key=$k2
  }
  $used[$key]=1
  $plan.Add([pscustomobject]@{ Origem=$f.FullName; Destino=$relDest; Nat=$nat; Tipo=$tipo; Materia=$mat; NomeNovo=$newName })
}
$plan | Export-Csv "$out\plano2.csv" -NoTypeInformation -Encoding UTF8

Write-Output ("`nArquivos no plano: {0:N0} | sem classificacao IA: {1:N0}" -f $plan.Count, $semCls)
Write-Output "`n=== Distribuicao por Nat\Tipo\Materia (top 30) ==="
$plan | Group-Object Nat, Tipo, Materia | Sort-Object Count -Descending | Select-Object -First 30 | ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Count, $_.Name) }
# quantos mudam de nome
$mud = 0; foreach($p in $plan){ if((Split-Path $p.Origem -Leaf) -ne $p.NomeNovo){ $mud++ } }
Write-Output ("`nArquivos que mudam de nome: {0:N0} de {1:N0}" -f $mud, $plan.Count)