# Investiga os .doc por tipo real de conteudo e verifica ferramentas de conversao disponiveis. Somente leitura.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'

Write-Output "=== Ferramentas de conversao ==="
$sofficeCand = @(
  "C:\Program Files\LibreOffice\program\soffice.exe",
  "C:\Program Files (x86)\LibreOffice\program\soffice.exe"
)
$soffice = $sofficeCand | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
Write-Output ("  LibreOffice (soffice): {0}" -f $(if($soffice){$soffice}else{'NAO encontrado'}))
$word = $null
try { $word = (New-Object -ComObject Word.Application); $wv = $word.Version; $word.Quit() | Out-Null; [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) } catch { $wv = $null }
Write-Output ("  Microsoft Word (COM): {0}" -f $(if($wv){"versao $wv"}else{'NAO disponivel'}))

function HeadHex($path, $n){
  try { $fs=[IO.File]::OpenRead($path); $buf=New-Object byte[] $n; $r=$fs.Read($buf,0,$n); $fs.Close(); return (($buf[0..([Math]::Max(0,$r-1))] | ForEach-Object { $_.ToString('X2') }) -join '') } catch { return '' }
}

Write-Output "`n=== Classificacao dos .doc por conteudo real ==="
$docs = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ieq '.doc' }
$cnt = @{ OLE=0; RTF=0; HTML_XML=0; ZIP_docx=0; TEXTO_OUTRO=0; VAZIO=0 }
$probList = New-Object System.Collections.Generic.List[object]
foreach($d in $docs){
  if($d.Length -eq 0){ $cnt.VAZIO++; continue }
  $h = HeadHex $d.FullName 8
  if($h -like 'D0CF11E0*'){ $cnt.OLE++ }
  elseif($h -like '7B5C7274*'){ $cnt.RTF++;       $probList.Add($d) }
  elseif($h -like '3C*' -or $h -like 'EFBBBF3C*'){ $cnt.HTML_XML++; $probList.Add($d) }
  elseif($h -like '504B0304*'){ $cnt.ZIP_docx++;  $probList.Add($d) }
  else { $cnt.TEXTO_OUTRO++; $probList.Add($d) }
}
$tot = ($docs | Measure-Object).Count
Write-Output ("  Total .doc: {0:N0}" -f $tot)
foreach($k in 'OLE','RTF','HTML_XML','ZIP_docx','TEXTO_OUTRO','VAZIO'){ Write-Output ("    {0,-12}: {1,6:N0}" -f $k, $cnt[$k]) }
Write-Output ("`n  .doc NAO-OLE (suspeitos de disparar dialogo de codificacao): {0:N0}" -f ($probList|Measure-Object).Count)
$probList | Select-Object @{n='Rel';e={$_.FullName.Substring($base.Length+1)}}, Length |
  Export-Csv "$out\doc_suspeitos_conversao.csv" -NoTypeInformation -Encoding UTF8
Write-Output "`n  Amostra de .doc nao-OLE:"
$probList | Select-Object -First 12 | ForEach-Object { Write-Output ("    {0}" -f $_.Name) }