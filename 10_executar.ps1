# EXECUTA o plano: apaga lixo (para a Lixeira) e move/renomeia conforme plano_organizacao.csv.
# Gera log de reversao. Use -DryRun para simular (padrao executa).
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$out  = 'C:\Users\mauri\ProjetoConversor'
$base = 'C:\Users\mauri\HD_Mau'
$plan = Import-Csv "$out\plano_organizacao.csv" -Encoding UTF8
$ts   = Get-Date -Format 'yyyyMMdd_HHmmss'
$log  = "$out\log_execucao_$ts.csv"

$logRows = New-Object System.Collections.Generic.List[object]
$okMove=0; $okDel=0; $erros=0; $errList = New-Object System.Collections.Generic.List[string]
$i=0
foreach($row in $plan){
  $i++
  try{
    if($row.Acao -eq 'APAGAR'){
      if(-not $DryRun){ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($row.Origem,'OnlyErrorDialogs','SendToRecycleBin') }
      $logRows.Add([pscustomobject]@{ Acao='APAGAR'; De=$row.Origem; Para='(Lixeira)' })
      $okDel++
    } else {
      $dir = [IO.Path]::GetDirectoryName($row.Destino)
      if(-not $DryRun){
        [IO.Directory]::CreateDirectory($dir) | Out-Null
        [IO.File]::Move($row.Origem, $row.Destino)
      }
      $logRows.Add([pscustomobject]@{ Acao='MOVER'; De=$row.Origem; Para=$row.Destino })
      $okMove++
    }
  } catch {
    $erros++
    if($errList.Count -lt 40){ $errList.Add("$($row.Origem) -> $($_.Exception.Message)") }
  }
  if($i % 5000 -eq 0){ Write-Output ("  ... processados {0:N0}/{1:N0}" -f $i, $plan.Count) }
}
$logRows | Export-Csv $log -NoTypeInformation -Encoding UTF8

# Remove diretorios vazios remanescentes (bottom-up), preservando a raiz
if(-not $DryRun){
  Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object {
      if((Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0){
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
      }
    }
}

Write-Output ("`n=== EXECUCAO {0} ===" -f $(if($DryRun){'(DRY-RUN)'}else{'CONCLUIDA'}))
Write-Output ("Movidos : {0:N0}" -f $okMove)
Write-Output ("Apagados: {0:N0} (Lixeira)" -f $okDel)
Write-Output ("Erros   : {0:N0}" -f $erros)
$errList | ForEach-Object { Write-Output ("  ERRO: {0}" -f $_) }
Write-Output ("Log de reversao: {0}" -f $log)