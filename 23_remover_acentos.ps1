# Remove acentos (transliteracao ASCII) dos nomes de ARQUIVOS e PASTAS de HD_Mau. Gera log. Trata colisoes.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'

function RemoveAccent($s){
  $n = $s.Normalize([Text.NormalizationForm]::FormD)
  $sb = New-Object System.Text.StringBuilder
  foreach($c in $n.ToCharArray()){
    if([Globalization.CharUnicodeInfo]::GetUnicodeCategory($c) -ne [Globalization.UnicodeCategory]::NonSpacingMark){ [void]$sb.Append($c) }
  }
  return $sb.ToString().Normalize([Text.NormalizationForm]::FormC)
}
function UniqueDest($dir, $name){
  $dest = Join-Path $dir $name
  if(-not (Test-Path -LiteralPath $dest)){ return $name }
  $bn = [IO.Path]::GetFileNameWithoutExtension($name); $ext = [IO.Path]::GetExtension($name); $i = 2
  do { $cand = "$bn ($i)$ext"; $dest = Join-Path $dir $cand; $i++ } while(Test-Path -LiteralPath $dest)
  return $cand
}

$log = New-Object System.Collections.Generic.List[object]
$okF=0; $okD=0; $err=0

# 1) ARQUIVOS
$files = Get-ChildItem -LiteralPath $base -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '00 - INDICE.md' }
foreach($f in $files){
  $novo = RemoveAccent $f.Name
  if($novo -ne $f.Name){
    try {
      $nn = UniqueDest $f.DirectoryName $novo
      Rename-Item -LiteralPath $f.FullName -NewName $nn -ErrorAction Stop
      $log.Add([pscustomobject]@{ Tipo='ARQ'; De=$f.FullName; Para=(Join-Path $f.DirectoryName $nn) }); $okF++
    } catch { $err++ }
  }
}

# 2) PASTAS (bottom-up: mais profundas primeiro)
$dirs = Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue | Sort-Object { ($_.FullName.ToCharArray() | Where-Object { $_ -eq '\' }).Count } -Descending
foreach($d in $dirs){
  $novo = RemoveAccent $d.Name
  if($novo -ne $d.Name){
    try {
      $parent = Split-Path -LiteralPath $d.FullName -Parent
      $nn = UniqueDest $parent $novo
      Rename-Item -LiteralPath $d.FullName -NewName $nn -ErrorAction Stop
      $log.Add([pscustomobject]@{ Tipo='DIR'; De=$d.FullName; Para=(Join-Path $parent $nn) }); $okD++
    } catch { $err++ }
  }
}
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$log | Export-Csv "$out\log_acentos_$ts.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Arquivos renomeados: {0:N0} | Pastas renomeadas: {1:N0} | Erros: {2}" -f $okF, $okD, $err)
Write-Output ("Log: $out\log_acentos_$ts.csv")