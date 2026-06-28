# Duplicatas LOGICAS: mesmo nome-base (ignorando " (N)", "- Copia") + extensao. Conteudo pode diferir. Somente leitura.
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$idx = Import-Csv "$out\indice_mestre.csv" -Encoding UTF8

function NormName($name){
  $b = [IO.Path]::GetFileNameWithoutExtension($name)
  $b = $b -replace '\s*\(\d+\)\s*$',''
  $b = $b -replace '\s*-\s*C[oó]pia.*$',''
  $b = $b -replace '^\s*C[oó]pia de\s*',''
  $b = $b -replace '\s*-\s*Copy.*$',''
  $b = $b -replace '\s*-\s*copy.*$',''
  return ($b.Trim().ToLower())
}

$grp = $idx | Group-Object { (NormName $_.Name) + '|' + $_.Ext } | Where-Object Count -gt 1
$arquivos = ($grp | Measure-Object Count -Sum).Sum
$excedentes = $arquivos - $grp.Count
[long]$bytesExced = ($grp | ForEach-Object { $_.Group | Sort-Object {[long]$_.Length} -Descending | Select-Object -Skip 1 } | Measure-Object Length -Sum).Sum

Write-Output ("Grupos de nome-base repetido : {0:N0}" -f $grp.Count)
Write-Output ("Arquivos envolvidos          : {0:N0}  (excedentes: {1:N0})" -f $arquivos, $excedentes)
Write-Output ("Tamanho dos excedentes       : {0:N2} GB  (NAO necessariamente apagavel - podem ser versoes)" -f ($bytesExced/1GB))

Write-Output "`n=== TOP 25 grupos com mais itens ==="
$grp | Sort-Object Count -Descending | Select-Object -First 25 | ForEach-Object {
  Write-Output ("  {0,3} x  {1}" -f $_.Count, ($_.Group[0].Name))
}
$grp | Select-Object @{n='NomeBase';e={(NormName $_.Group[0].Name)}}, Count,
  @{n='Itens';e={ ($_.Group | ForEach-Object { $_.Rel }) -join ' || ' }} |
  Sort-Object Count -Descending | Export-Csv "$out\dup_logicas.csv" -NoTypeInformation -Encoding UTF8
Write-Output "`nDetalhe salvo em $out\dup_logicas.csv"