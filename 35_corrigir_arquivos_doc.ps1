# Corrige a classificacao de ARQUIVOS: usa a categoria do CSV (UTF-8) como verdade. Sem literais acentuados no script.
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cls = Import-Csv "$out\arquivos_class.csv" -Delimiter '|' -Encoding UTF8
$catDirs = @(Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -match '^\d\d - ' })
$ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_doc_corrige_$ts.csv"
$rows=New-Object System.Collections.Generic.List[object]; $ok=0;$ja=0;$nf=0
foreach($c in $cls){
  $destCat = Join-Path $base $c.categoria
  $dest = Join-Path $destCat $c.arquivo
  if(Test-Path -LiteralPath $dest){ $ja++; continue }   # ja no lugar certo
  # achar o arquivo atual em alguma categoria (nome exato ou variacao (N))
  $atual = $null
  foreach($cd in $catDirs){
    $cand = Join-Path $cd.FullName $c.arquivo
    if(Test-Path -LiteralPath $cand){ $atual=$cand; break }
  }
  if(-not $atual){
    # tentar variacao com sufixo (procura por base do nome)
    $bn=[IO.Path]::GetFileNameWithoutExtension($c.arquivo); $ext=[IO.Path]::GetExtension($c.arquivo)
    foreach($cd in $catDirs){
      $m = Get-ChildItem -LiteralPath $cd.FullName -File -Force -EA SilentlyContinue | Where-Object { $_.Name -like "$bn ``(*``)$ext" } | Select-Object -First 1
      if($m){ $atual=$m.FullName; break }
    }
  }
  if(-not $atual){ $nf++; continue }
  if(-not (Test-Path -LiteralPath $destCat)){ [IO.Directory]::CreateDirectory($destCat)|Out-Null }
  $d=$dest
  if(Test-Path -LiteralPath $d){ $bn=[IO.Path]::GetFileNameWithoutExtension($c.arquivo);$ext=[IO.Path]::GetExtension($c.arquivo);$i=2; do{ $d=Join-Path $destCat "$bn ($i)$ext";$i++ }while(Test-Path -LiteralPath $d) }
  try { [IO.File]::Move($atual,$d); $rows.Add([pscustomobject]@{De=$atual;Para=$d}); $ok++ } catch {}
}
$rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
Write-Output ("Corrigidos (movidos p/ cat certa): {0} | ja corretos: {1} | nao encontrados: {2}" -f $ok,$ja,$nf)
Write-Output ("Log: {0}" -f $log)