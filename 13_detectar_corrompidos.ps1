# Detecta arquivos temporarios, vazios e corrompidos (alta confianca). Gera indice atual + lista de suspeitos. NAO exclui.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$prefixLen = $base.Length + 1

function HeadBytes($path, $n){
  try { $fs=[IO.File]::OpenRead($path); $buf=New-Object byte[] $n; $r=$fs.Read($buf,0,$n); $fs.Close(); if($r -lt $n){ return $buf[0..([Math]::Max(0,$r-1))] } else { return $buf } }
  catch { return $null }
}
function ToHex($bytes){ if($bytes){ ($bytes | ForEach-Object { $_.ToString('X2') }) -join '' } else { '' } }

$tempPat = '^~\$|\.tmp$|\.temp$|\.part$|\.crdownload$|\.partial$|^thumbs\.db$|^desktop\.ini$|\.ds_store$|^~wrl.*\.tmp$'
$idx = New-Object System.Collections.Generic.List[object]
$susp = New-Object System.Collections.Generic.List[object]

$files = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '00 - INDICE.md' }
$nfiles = ($files | Measure-Object).Count
$i = 0
foreach($f in $files){
  $i++
  $rel = $f.FullName.Substring($prefixLen)
  $ext = $f.Extension.ToLower()
  $len = [long]$f.Length
  $idx.Add([pscustomobject]@{ Rel=$rel; Name=$f.Name; Ext=$ext; Length=$len; FullName=$f.FullName })

  $motivo = $null
  if($f.Name -match $tempPat){ $motivo = 'TEMPORARIO' }
  elseif($len -eq 0){ $motivo = 'VAZIO (0 bytes)' }
  elseif($ext -eq '.pdf'){
    $h = HeadBytes $f.FullName 1024
    if($h -eq $null){ $motivo = 'ERRO_LEITURA' }
    elseif((ToHex $h) -notlike '*25504446*'){ $motivo = 'PDF sem assinatura %PDF' }
  }
  elseif($ext -in '.docx','.xlsx','.pptx'){
    $h = HeadBytes $f.FullName 4
    if($h -eq $null){ $motivo = 'ERRO_LEITURA' }
    elseif((ToHex $h) -notlike '504B*'){ $motivo = 'Office novo sem assinatura ZIP' }
  }
  if($motivo){ $susp.Add([pscustomobject]@{ Motivo=$motivo; Rel=$rel; Ext=$ext; Length=$len; FullName=$f.FullName }) }
}
$idx  | Export-Csv "$out\indice_atual.csv" -NoTypeInformation -Encoding UTF8
$susp | Export-Csv "$out\suspeitos_exclusao.csv" -NoTypeInformation -Encoding UTF8

Write-Output ("Arquivos analisados: {0:N0}" -f $nfiles)
Write-Output ("`n=== Suspeitos por motivo ===")
$susp | Group-Object Motivo | Sort-Object Count -Descending | ForEach-Object { Write-Output ("  {0,5:N0}  {1}" -f $_.Count, $_.Name) }
Write-Output ("`nTOTAL suspeitos: {0:N0}" -f ($susp|Measure-Object).Count)
Write-Output "`n=== Amostra (ate 25) ==="
$susp | Select-Object -First 25 | ForEach-Object { Write-Output ("  [{0}] {1}" -f $_.Motivo, (Split-Path $_.Rel -Leaf)) }