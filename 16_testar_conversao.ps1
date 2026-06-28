# Lista os .doc nao-OLE atuais e testa a conversao de 1 deles para .docx via Word COM.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
function HeadHex($path,$n){ try { $fs=[IO.File]::OpenRead($path); $buf=New-Object byte[] $n; $r=$fs.Read($buf,0,$n); $fs.Close(); return (($buf[0..([Math]::Max(0,$r-1))]|ForEach-Object{$_.ToString('X2')}) -join '') } catch { return '' } }

$docs = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ieq '.doc' -and $_.Length -gt 0 }
$naoOLE = @($docs | Where-Object { (HeadHex $_.FullName 8) -notlike 'D0CF11E0*' })
Write-Output ("`.doc nao-OLE reais (apos limpeza): {0}" -f $naoOLE.Count)
$naoOLE | Select-Object @{n='Rel';e={$_.FullName.Substring($base.Length+1)}}, Length | Export-Csv "$out\doc_converter.csv" -NoTypeInformation -Encoding UTF8
$naoOLE | ForEach-Object { Write-Output ("   {0}" -f $_.Name) }

if($naoOLE.Count -eq 0){ Write-Output "Nada a converter."; return }

$alvo = $naoOLE[0]
Write-Output ("`nTestando conversao de: {0}" -f $alvo.Name)
$sw = [Diagnostics.Stopwatch]::StartNew()
$w = $null
try {
  $w = New-Object -ComObject Word.Application
  $w.Visible = $false
  $w.DisplayAlerts = 0
  $w.Options.ConfirmConversions = $false
  $doc = $w.Documents.Open($alvo.FullName, $false, $true)   # ConfirmConversions=false, ReadOnly=true
  $dest = [IO.Path]::ChangeExtension($alvo.FullName, '.docx')
  $doc.SaveAs2($dest, 16)   # 16 = wdFormatXMLDocument (.docx)
  $doc.Close($false)
  $sw.Stop()
  $okHex = (HeadHex $dest 4)
  Write-Output ("  OK em {0:N1}s -> {1} (assinatura {2}; valido={3})" -f $sw.Elapsed.TotalSeconds, (Split-Path $dest -Leaf), $okHex, ($okHex -like '504B*'))
} catch {
  Write-Output ("  FALHOU: {0}" -f $_.Exception.Message)
} finally {
  if($w){ try { $w.Quit() } catch {} ; [void][Runtime.InteropServices.Marshal]::ReleaseComObject($w) }
}