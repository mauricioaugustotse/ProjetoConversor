# Analise de padroes de nome para fundamentar o esquema de organizacao (somente leitura).
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\mauri\ProjetoConversor'
$idx = Import-Csv "$out\indice_mestre.csv" -Encoding UTF8

function Pref($name){
  $b = [IO.Path]::GetFileNameWithoutExtension($name)
  if($b -match ' - '){ return (($b -split ' - ',2)[0]).Trim() }
  return $b.Trim()
}
function Tok0($name){
  $b = [IO.Path]::GetFileNameWithoutExtension($name).Trim()
  return (($b -split '[\s\-_]+',2)[0])
}

Write-Output "=== TOP 45 PREFIXOS (texto antes do 1o ' - ') ==="
$idx | Group-Object { Pref $_.Name } | Sort-Object Count -Descending | Select-Object -First 45 |
  ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Count, $_.Name) }

Write-Output "`n=== TOP 45 PRIMEIRA PALAVRA/SIGLA ==="
$idx | Group-Object { Tok0 $_.Name } | Sort-Object Count -Descending | Select-Object -First 45 |
  ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Count, $_.Name) }

# Palavras-chave tematicas frequentes (token simples)
$kw = 'acórdão|acordao|sentença|sentenca|petição|peticao|despacho|decisão|decisao|parecer|contrato|ofício|oficio|memorando|ata|edital|laudo|recurso|mandado|certidão|certidao|procuração|procuracao|relatório|relatorio|portaria|nota|voto|liminar|agravo|embargos|apelação|apelacao|contestação|contestacao'
Write-Output "`n=== Ocorrencia de PALAVRAS-CHAVE tematicas no nome ==="
$tot = @{}
foreach($f in $idx){
  $n = $f.Name.ToLower()
  foreach($k in ($kw -split '\|')){ if($n -like "*$k*"){ if($tot.ContainsKey($k)){$tot[$k]++}else{$tot[$k]=1} } }
}
$tot.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object { Write-Output ("  {0,6:N0}  {1}" -f $_.Value, $_.Name) }