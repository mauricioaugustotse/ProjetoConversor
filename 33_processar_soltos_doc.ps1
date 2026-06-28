# Processa os arquivos soltos da raiz de Documentos: remove duplicatas (Lixeira) e move/renomeia o resto.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cfg = Get-Content -Raw -Encoding UTF8 "$out\doc_soltos.json" | ConvertFrom-Json

# validacao
$faltam=@()
foreach($r in $cfg.remover){ if(-not (Test-Path -LiteralPath (Join-Path $base $r))){ $faltam += "REMOVER: $r" } }
foreach($m in $cfg.mover){ if(-not (Test-Path -LiteralPath (Join-Path $base $m.f))){ $faltam += "MOVER: $($m.f)" } }
Write-Output ("Remover: {0} | Mover: {1} | nao encontrados: {2}" -f $cfg.remover.Count, $cfg.mover.Count, $faltam.Count)
$faltam | ForEach-Object { Write-Output "  $_" }

if($Execute -and $faltam.Count -eq 0){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_doc_soltos_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $del=0;$mov=0;$err=0
  foreach($r in $cfg.remover){
    $p=Join-Path $base $r
    try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p,'OnlyErrorDialogs','SendToRecycleBin'); $rows.Add([pscustomobject]@{Acao='REMOVIDO(Lixeira)';De=$p;Para=''}); $del++ } catch { $err++ }
  }
  foreach($m in $cfg.mover){
    $src=Join-Path $base $m.f
    $catDir=Join-Path $base $m.cat
    if(-not (Test-Path -LiteralPath $catDir)){ [IO.Directory]::CreateDirectory($catDir)|Out-Null }
    $nome = if($m.novo){ $m.novo } else { $m.f }
    $dst=Join-Path $catDir $nome
    if(Test-Path -LiteralPath $dst){ $bn=[IO.Path]::GetFileNameWithoutExtension($nome);$ext=[IO.Path]::GetExtension($nome);$i=2; do{ $dst=Join-Path $catDir "$bn ($i)$ext";$i++ }while(Test-Path -LiteralPath $dst) }
    try { [IO.File]::Move($src,$dst); $rows.Add([pscustomobject]@{Acao='MOVIDO';De=$src;Para=$dst}); $mov++ } catch { $err++ }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRemovidos: {0} | Movidos: {1} | Erros: {2} | Log: {3}" -f $del,$mov,$err,$log)
}