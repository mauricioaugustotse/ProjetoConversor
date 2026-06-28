# Converte os .doc nao-OLE reais para .docx (Word COM) e exclui o .doc original; remove os ~WRD temporarios.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
function HeadHex($path,$n){ try { $fs=[IO.File]::OpenRead($path); $buf=New-Object byte[] $n; $r=$fs.Read($buf,0,$n); $fs.Close(); return (($buf[0..([Math]::Max(0,$r-1))]|ForEach-Object{$_.ToString('X2')}) -join '') } catch { return '' } }

$docs = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ieq '.doc' -and $_.Length -gt 0 }
$naoOLE = @($docs | Where-Object { (HeadHex $_.FullName 8) -notlike 'D0CF11E0*' })
$wrd  = @($naoOLE | Where-Object { $_.Name -match '^~WRD' })
$reais = @($naoOLE | Where-Object { $_.Name -notmatch '^~WRD' })
Write-Output ("nao-OLE: {0} | ~WRD temporarios: {1} | reais a converter: {2}" -f $naoOLE.Count, $wrd.Count, $reais.Count)

# Excluir ~WRD temporarios
foreach($t in $wrd){ Remove-Item -LiteralPath $t.FullName -Force -ErrorAction SilentlyContinue }
Write-Output ("Temporarios ~WRD removidos: {0}" -f $wrd.Count)

# Converter reais
$log = New-Object System.Collections.Generic.List[object]
$ok=0; $err=0
$w = $null
try {
  $w = New-Object -ComObject Word.Application
  $w.Visible = $false; $w.DisplayAlerts = 0; $w.Options.ConfirmConversions = $false
  foreach($f in $reais){
    $dest = [IO.Path]::ChangeExtension($f.FullName, '.docx')
    try {
      if((Test-Path -LiteralPath $dest) -and ((HeadHex $dest 4) -like '504B*')){
        # ja convertido (ex.: teste anterior) -> apenas remover o .doc original
      } else {
        $doc = $w.Documents.Open($f.FullName, $false, $true)
        $doc.SaveAs2($dest, 16)
        $doc.Close($false)
      }
      if((Test-Path -LiteralPath $dest) -and ((HeadHex $dest 4) -like '504B*')){
        Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
        $ok++; $log.Add([pscustomobject]@{ Origem=$f.FullName; Destino=$dest; Status='OK' })
      } else { $err++; $log.Add([pscustomobject]@{ Origem=$f.FullName; Destino=$dest; Status='FALHA_VALIDACAO' }) }
    } catch { $err++; $log.Add([pscustomobject]@{ Origem=$f.FullName; Destino=$dest; Status=("ERRO: "+$_.Exception.Message) }) }
  }
} finally { if($w){ try { $w.Quit() } catch {} ; [void][Runtime.InteropServices.Marshal]::ReleaseComObject($w) } }
$log | Export-Csv "$out\conversao_doc_log.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("`nConvertidos para .docx: {0} | Erros: {1}" -f $ok, $err)
$log | Where-Object { $_.Status -ne 'OK' } | ForEach-Object { Write-Output ("  {0} -> {1}" -f (Split-Path $_.Origem -Leaf), $_.Status) }