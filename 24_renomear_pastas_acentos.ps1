# Renomeia PASTAS (e arquivos restantes) removendo acentos. Bottom-up com passadas. Idempotente.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
function RemoveAccent($s){ $n=$s.Normalize([Text.NormalizationForm]::FormD); $sb=New-Object System.Text.StringBuilder; foreach($c in $n.ToCharArray()){ if([Globalization.CharUnicodeInfo]::GetUnicodeCategory($c) -ne [Globalization.UnicodeCategory]::NonSpacingMark){ [void]$sb.Append($c) } }; return $sb.ToString().Normalize([Text.NormalizationForm]::FormC) }
function UniqueDest($dir,$name){ $d=Join-Path $dir $name; if(-not(Test-Path -LiteralPath $d)){return $name}; $bn=[IO.Path]::GetFileNameWithoutExtension($name);$ext=[IO.Path]::GetExtension($name);$i=2; do{ $c="$bn ($i)$ext";$d=Join-Path $dir $c;$i++ }while(Test-Path -LiteralPath $d); return $c }

$totD=0
for($pass=0;$pass -lt 12;$pass++){
  $comAc = @(Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue | Where-Object { (RemoveAccent $_.Name) -ne $_.Name } | Sort-Object { ($_.FullName -split '\\').Count } -Descending)
  if($comAc.Count -eq 0){ break }
  foreach($d in $comAc){
    if(-not (Test-Path -LiteralPath $d.FullName)){ continue }
    $parent = Split-Path -LiteralPath $d.FullName -Parent
    $nn = UniqueDest $parent (RemoveAccent $d.Name)
    try { Rename-Item -LiteralPath $d.FullName -NewName $nn -ErrorAction Stop; $totD++ } catch { Write-Output ("ERRO DIR: {0} -> {1}" -f $d.Name, $_.Exception.Message) }
  }
}
$totF=0
$comAcF = @(Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '00 - INDICE.md' -and (RemoveAccent $_.Name) -ne $_.Name })
foreach($f in $comAcF){
  $nn = UniqueDest $f.DirectoryName (RemoveAccent $f.Name)
  try { Rename-Item -LiteralPath $f.FullName -NewName $nn -ErrorAction Stop; $totF++ } catch { Write-Output ("ERRO ARQ: {0} -> {1}" -f $f.Name, $_.Exception.Message) }
}
Write-Output ("Pastas renomeadas: {0} | Arquivos restantes renomeados: {1}" -f $totD, $totF)
$remD = @(Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue | Where-Object { (RemoveAccent $_.Name) -ne $_.Name }).Count
$remF = @(Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '00 - INDICE.md' -and (RemoveAccent $_.Name) -ne $_.Name }).Count
Write-Output ("Remanescentes com acento -> pastas: {0} | arquivos: {1}" -f $remD, $remF)