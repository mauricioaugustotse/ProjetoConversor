# Exclui os placeholders online-only orfaos da pasta TSE. Registra a lista antes de excluir.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
$tse  = "$base\OneDrive - TRIBUNAL SUPERIOR ELEITORAL"
$out  = 'C:\Users\mauri\ProjetoConversor'

if(-not (Test-Path -LiteralPath $tse)){ Write-Output "Pasta TSE nao existe (nada a fazer)."; return }

# 1) Registro permanente do que sera excluido
$files = Get-ChildItem -LiteralPath $tse -Recurse -File -Force -ErrorAction SilentlyContinue
$files | Select-Object @{n='Rel';e={$_.FullName.Substring($base.Length+1)}}, Name, Length,
  @{n='Modified';e={$_.LastWriteTime.ToString('yyyy-MM-dd')}},
  @{n='Attributes';e={$_.Attributes.ToString()}} |
  Export-Csv "$out\tse_placeholders_excluidos.csv" -NoTypeInformation -Encoding UTF8
$total = ($files | Measure-Object).Count
[long]$bytesLog = ($files | Measure-Object Length -Sum).Sum
Write-Output ("Placeholders registrados: {0:N0}  (tamanho logico {1:N2} GB)" -f $total, ($bytesLog/1GB))
Write-Output "Registro salvo em: $out\tse_placeholders_excluidos.csv"

# 2) Exclusao (Remove-Item direto: nao materializa o stub, nao usa UI)
$okDel = 0; $err = 0; $errs = New-Object System.Collections.Generic.List[string]
foreach($f in $files){
  try { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop; $okDel++ }
  catch { $err++; if($errs.Count -lt 20){ $errs.Add(("{0} -> {1}" -f $f.Name, $_.Exception.Message)) } }
}
Write-Output ("`nExcluidos: {0:N0} | Erros: {1:N0}" -f $okDel, $err)
$errs | ForEach-Object { Write-Output "  ERRO: $_" }

# 3) Remove subpastas vazias e a propria pasta TSE
Get-ChildItem -LiteralPath $tse -Recurse -Directory -Force -ErrorAction SilentlyContinue |
  Sort-Object { $_.FullName.Length } -Descending |
  ForEach-Object {
    if((Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0){
      Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }
  }
if(Test-Path -LiteralPath $tse){
  $rest = (Get-ChildItem -LiteralPath $tse -Force -ErrorAction SilentlyContinue | Measure-Object).Count
  if($rest -eq 0){ Remove-Item -LiteralPath $tse -Force -ErrorAction SilentlyContinue; Write-Output "`nPasta TSE removida (vazia)." }
  else { Write-Output ("`nPasta TSE ainda contem {0} item(ns) (ex.: desktop.ini do OneDrive); nao removida." -f $rest) }
}