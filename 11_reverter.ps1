# DESFAZ a organizacao: le o log de execucao mais recente e move tudo de volta para a origem.
# (Os arquivos APAGADOS estao na Lixeira do Windows e devem ser restaurados manualmente de la.)
param([string]$LogFile)
$ErrorActionPreference = 'Stop'
$out = 'C:\Users\mauri\ProjetoConversor'
if(-not $LogFile){
  $LogFile = (Get-ChildItem "$out\log_execucao_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
Write-Output "Revertendo a partir de: $LogFile"
$rows = Import-Csv $LogFile -Encoding UTF8 | Where-Object Acao -eq 'MOVER'
$ok=0; $erros=0
# Reverter em ordem inversa
[array]::Reverse($rows)
foreach($r in $rows){
  try{
    $dir = [IO.Path]::GetDirectoryName($r.De)
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    [IO.File]::Move($r.Para, $r.De)
    $ok++
  } catch { $erros++ }
}
Write-Output ("Revertidos: {0:N0} | Erros: {1:N0}" -f $ok, $erros)
Write-Output "OBS: arquivos enviados a Lixeira nao sao restaurados por este script (restaure pela Lixeira do Windows)."