# Limpeza do TSE: remove o backup duplicado (3,9 GB) e as pastas *_files (paginas salvas). Tudo para a Lixeira.
$ErrorActionPreference='Continue'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'
$catJur=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '05 *' } | Select-Object -First 1
$tse=Join-Path $catJur.FullName 'TSE'
Write-Output ("Pasta TSE: {0}" -f $tse)

# 1) Backup duplicado
$bk="$tse\ONE_DRIVE_backup"
if(Test-Path -LiteralPath $bk){
  $sz=(Get-ChildItem -LiteralPath $bk -Recurse -File -Force -EA SilentlyContinue|Measure-Object Length -Sum).Sum
  try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($bk,'OnlyErrorDialogs','SendToRecycleBin'); Write-Output ("Backup ONE_DRIVE_backup removido p/ Lixeira ({0:N1} GB)" -f ($sz/1GB)) }
  catch { Write-Output ("ERRO ao remover backup: {0}" -f $_.Exception.Message) }
} else { Write-Output "ONE_DRIVE_backup nao encontrado" }

# 2) Pastas *_files restantes (paginas salvas) - mantem o .html irmao
$rm=0; $rmMB=0
foreach($f in (Get-ChildItem -LiteralPath $tse -Recurse -Directory -Force -EA SilentlyContinue | Where-Object {$_.Name -like '*_files'})){
  if(-not (Test-Path -LiteralPath $f.FullName)){ continue }
  $sz=(Get-ChildItem -LiteralPath $f.FullName -Recurse -File -Force -EA SilentlyContinue|Measure-Object Length -Sum).Sum
  try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($f.FullName,'OnlyErrorDialogs','SendToRecycleBin'); $rm++; $rmMB+=($sz/1MB) } catch {}
}
Write-Output ("Pastas _files removidas: {0} ({1:N1} MB)" -f $rm, $rmMB)