# Executa o plano2 (move/renomeia para subpastas por materia com nome novo). Gera log de reversao. -DryRun simula.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$plan = Import-Csv "$out\plano2.csv" -Encoding UTF8
$ts   = Get-Date -Format 'yyyyMMdd_HHmmss'
$log  = "$out\log_execucao2_$ts.csv"
$logRows = New-Object System.Collections.Generic.List[object]
$ok=0; $err=0; $skip=0; $errs = New-Object System.Collections.Generic.List[string]
$i=0
foreach($r in $plan){
  $i++
  try {
    if(-not (Test-Path -LiteralPath $r.Origem)){ $skip++; continue }
    if($r.Origem -eq $r.Destino){ $skip++; continue }
    $dir = [IO.Path]::GetDirectoryName($r.Destino)
    if(-not $DryRun){
      [IO.Directory]::CreateDirectory($dir) | Out-Null
      [IO.File]::Move($r.Origem, $r.Destino)
    }
    $logRows.Add([pscustomobject]@{ De=$r.Origem; Para=$r.Destino }); $ok++
  } catch { $err++; if($errs.Count -lt 40){ $errs.Add("$($r.Origem) -> $($_.Exception.Message)") } }
  if($i % 5000 -eq 0){ Write-Output ("  ... {0:N0}/{1:N0}" -f $i, $plan.Count) }
}
$logRows | Export-Csv $log -NoTypeInformation -Encoding UTF8

if(-not $DryRun){
  Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { if((Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0){ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } }
}
Write-Output ("`n=== EXECUCAO2 {0} ===" -f $(if($DryRun){'(DRY-RUN)'}else{'CONCLUIDA'}))
Write-Output ("Movidos: {0:N0} | Pulados: {1:N0} | Erros: {2:N0}" -f $ok, $skip, $err)
$errs | ForEach-Object { Write-Output ("  ERRO: {0}" -f $_) }
Write-Output ("Log de reversao: {0}" -f $log)