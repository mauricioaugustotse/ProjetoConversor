# Estima .doc de versoes antigas do Word lendo o wIdent (offset 512). ECA5 = Word 97-2003 (moderno). Somente leitura.
$ErrorActionPreference = 'SilentlyContinue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$docs = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ieq '.doc' -and $_.Length -gt 514 }
$w97 = 0; $susp = New-Object System.Collections.Generic.List[object]
foreach($d in $docs){
  try {
    $fs = [IO.File]::OpenRead($d.FullName)
    $null = $fs.Seek(512, [IO.SeekOrigin]::Begin)
    $b = New-Object byte[] 2; $null = $fs.Read($b, 0, 2); $fs.Close()
    $hex = ('{0:X2}{1:X2}' -f $b[0], $b[1])
    if($hex -eq 'ECA5'){ $w97++ } else { $susp.Add([pscustomobject]@{ Rel=$d.FullName.Substring($base.Length+1); WIdent=$hex }) }
  } catch {}
}
$susp | Export-Csv "$out\doc_possiveis_antigos.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Total .doc OLE analisados: {0:N0}" -f ($docs|Measure-Object).Count)
Write-Output ("  Word 97-2003 (wIdent ECA5 em offset 512): {0:N0}" -f $w97)
Write-Output ("  Sem ECA5 nesse offset (antigos OU stream em outro local): {0:N0}" -f ($susp|Measure-Object).Count)
Write-Output "`nDistribuicao dos wIdent encontrados (top):"
$susp | Group-Object WIdent | Sort-Object Count -Descending | Select-Object -First 10 | ForEach-Object { Write-Output ("  {0}: {1:N0}" -f $_.Name, $_.Count) }