# Processa o texto extraido (extraido.tsv): regex de metadados + deteccao de tipo/materia por conteudo + dedup. Gera relatorio. SEM IA.
$ErrorActionPreference = 'SilentlyContinue'
$out  = 'C:\Users\mauri\ProjetoConversor'
$base = 'C:\Users\mauri\HD_Mau'
$tsv  = if($args[0]){ $args[0] } else { "$out\conteudo\extraido.tsv" }

$rows = New-Object System.Collections.Generic.List[object]
$nLinhas = 0
foreach($line in [IO.File]::ReadLines($tsv, [Text.Encoding]::UTF8)){
  $nLinhas++
  $p = $line -split "`t", 3
  if($p.Count -lt 2){ continue }
  $path = $p[0]; $len = [int]$p[1]; $txt = if($p.Count -ge 3){ $p[2] } else { '' }
  $segs = $path -split '/'
  $idxb = -1
  for($si=0; $si -lt $segs.Count; $si++){ if($segs[$si].ToLower() -eq 'hd_mau'){ $idxb=$si; break } }
  if($idxb -ge 0 -and ($idxb+1) -lt $segs.Count){ $natAtual = $segs[$idxb+1]; $rel = ($segs[($idxb+1)..($segs.Count-1)] -join '\') } else { $natAtual=''; $rel=$path }

  # versao "comprimida" p/ CNJ (antiword/pdftotext metem espacos entre digitos)
  $comp = ($txt -replace '\s', '')
  $cnj = ''
  $m = [regex]::Match($comp, '\d{7}-?\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}')
  if($m.Success){ $cnj = $m.Value }

  $U = $txt.ToUpperInvariant()
  # deteccao de tipo por conteudo (radicais ASCII p/ robustez a encoding)
  $tipo = ''
  if($U -match 'CURSOS? ON-?LINE|PONTO DOS CONCURSOS|\bAULA\s*0*\d|ESQUEMATIZAD|APOSTILA|MATERIAL DID|CADERNO DE QUEST|VADE ?MECUM'){ $tipo='Estudo' }
  elseif($U -match '\bEMENTA\b|ACORDAM|AC[OÓ]RD[AÃ]O'){ $tipo='Acordao' }
  elseif($U -match '\bDESPACHO\b'){ $tipo='Despacho' }
  elseif($U -match 'DECIS[AÃ]O'){ $tipo='Decisao' }
  elseif($U -match 'PETI[CÇ][AÃ]O'){ $tipo='Peticao' }
  elseif($U -match 'CONTESTA[CÇ]|MANIFESTA[CÇ]'){ $tipo='Contestacao' }
  elseif($U -match '\bEDITAL\b'){ $tipo='Edital' }
  elseif($U -match '\bOF[IÍ]CIO\b|MEMORANDO'){ $tipo='Oficio' }
  elseif($U -match 'RECURSO|AGRAVO|EMBARGOS|APELA[CÇ]'){ $tipo='Recurso' }

  # tribunal/orgao
  $org = ''
  if($U -match 'TRIBUNAL SUPERIOR ELEITORAL|\bTSE\b'){ $org='TSE' }
  elseif($U -match 'TRIBUNAL REGIONAL ELEITORAL|\bTRE'){ $org='TRE' }
  elseif($U -match '\bTST\b|TRIBUNAL SUPERIOR DO TRABALHO'){ $org='TST' }
  elseif($U -match '\bSTJ\b'){ $org='STJ' }
  elseif($U -match '\bSTF\b|SUPREMO'){ $org='STF' }

  $rows.Add([pscustomobject]@{ Rel=$rel; NatAtual=$natAtual; Len=$len; CNJ=$cnj; TipoConteudo=$tipo; Orgao=$org; Texto=$txt })
}

# Dedup por conteudo (hash do texto normalizado, ignora vazios)
$md5 = [Security.Cryptography.MD5]::Create()
$byHash = @{}
foreach($r in $rows){
  if($r.Len -lt 40){ continue }
  $norm = ($r.Texto.ToLowerInvariant() -replace '[^a-z0-9]','')
  if($norm.Length -lt 30){ continue }
  $h = [BitConverter]::ToString($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($norm)))
  if(-not $byHash.ContainsKey($h)){ $byHash[$h]=0 }; $byHash[$h]++
}
$dupGrupos = @($byHash.Values | Where-Object { $_ -gt 1 }).Count
$dupExced  = ($byHash.Values | Where-Object { $_ -gt 1 } | ForEach-Object { $_-1 } | Measure-Object -Sum).Sum

# Metricas
$comTexto = @($rows | Where-Object { $_.Len -ge 80 }).Count
$semTexto = @($rows | Where-Object { $_.Len -lt 80 }).Count
$comCNJ   = @($rows | Where-Object { $_.CNJ -ne '' }).Count
$estudoEmDiversos = @($rows | Where-Object { $_.NatAtual -like '05*' -and $_.TipoConteudo -eq 'Estudo' }).Count
$diversosComTexto = @($rows | Where-Object { $_.NatAtual -like '05*' -and $_.Len -ge 80 }).Count
$diversosSemTexto = @($rows | Where-Object { $_.NatAtual -like '05*' -and $_.Len -lt 80 }).Count

$rows | Select-Object Rel,NatAtual,Len,CNJ,TipoConteudo,Orgao | Export-Csv "$out\conteudo_metadados.csv" -NoTypeInformation -Encoding UTF8

Write-Output ("Documentos com texto lido: {0}" -f $nLinhas)
Write-Output ("  Com texto util (>=80 chars): {0:N0}" -f $comTexto)
Write-Output ("  SEM texto (escaneados/imagem): {0:N0}" -f $semTexto)
Write-Output ("  Com numero de processo CNJ detectado: {0:N0}" -f $comCNJ)
Write-Output ("  Tipo detectado por conteudo:")
$rows | Group-Object TipoConteudo | Sort-Object Count -Descending | ForEach-Object { Write-Output ("     {0,6:N0}  {1}" -f $_.Count, $(if($_.Name){$_.Name}else{'(indefinido)'})) }
Write-Output ("`n  Duplicatas por CONTEUDO: {0:N0} grupos / {1:N0} excedentes" -f $dupGrupos, $dupExced)
Write-Output ("`n  Pasta 05-Diversos: {0:N0} com texto | {1:N0} sem texto (escaneados)" -f $diversosComTexto, $diversosSemTexto)
Write-Output ("     - destes, materiais de ESTUDO detectados (reclassificaveis de graca): {0:N0}" -f $estudoEmDiversos)
Write-Output "`nMetadados salvos em conteudo_metadados.csv"