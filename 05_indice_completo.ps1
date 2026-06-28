# Indice mestre COMPLETO de HD_Mau (inclui OneDrive TSE). Somente metadados.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$tse  = Join-Path $base 'OneDrive - TRIBUNAL SUPERIOR ELEITORAL'
$prefixLen = $base.Length + 1

$files = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue |
  Select-Object @{n='Rel';e={$_.FullName.Substring($prefixLen)}}, Name,
                @{n='Ext';e={$_.Extension.ToLower()}}, Length,
                @{n='Modified';e={$_.LastWriteTime.ToString('yyyy-MM-dd')}},
                @{n='IsTSE';e={ if($_.FullName -like "$tse*"){1}else{0} }}, FullName
$files | Export-Csv "$out\indice_mestre.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Indice mestre COMPLETO: {0:N0} arquivos" -f $files.Count)
Write-Output ("  Na raiz (fora TSE): {0:N0}" -f ($files | Where-Object IsTSE -eq 0).Count)
Write-Output ("  Na pasta TSE      : {0:N0}" -f ($files | Where-Object IsTSE -eq 1).Count)
Write-Output "Salvo em $out\indice_mestre.csv"