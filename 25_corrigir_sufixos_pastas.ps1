# Remove sufixos espurios " (N)" dos nomes de PASTAS (nao toca em arquivos). Bottom-up. Mescla se houver colisao real.
$ErrorActionPreference = 'Continue'
$base = 'C:\Users\mauri\HD_Mau'
function StripSuffix($name){ if($name -match '^(.*) \(\d+\)$'){ return $matches[1] } else { return $name } }

$ren=0; $merge=0
for($pass=0; $pass -lt 15; $pass++){
  $alvos = @(Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match ' \(\d+\)$' } |
    Sort-Object { ($_.FullName -split '\\').Count } -Descending)
  if($alvos.Count -eq 0){ break }
  foreach($d in $alvos){
    if(-not (Test-Path -LiteralPath $d.FullName)){ continue }
    $parent = [IO.Path]::GetDirectoryName($d.FullName)
    $novo = StripSuffix $d.Name
    $destPath = Join-Path $parent $novo
    if(-not (Test-Path -LiteralPath $destPath)){
      try { Rename-Item -LiteralPath $d.FullName -NewName $novo -ErrorAction Stop; $ren++ } catch { Write-Output ("ERRO rename: {0} -> {1}" -f $d.Name, $_.Exception.Message) }
    } else {
      # colisao real: mover conteudo para a pasta base e remover a duplicada
      Get-ChildItem -LiteralPath $d.FullName -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $tgt = Join-Path $destPath $_.Name
        if(Test-Path -LiteralPath $tgt){ $tgt = Join-Path $destPath ([IO.Path]::GetFileNameWithoutExtension($_.Name) + ' (m)' + [IO.Path]::GetExtension($_.Name)) }
        try { Move-Item -LiteralPath $_.FullName -Destination $tgt -ErrorAction Stop } catch {}
      }
      try { Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction Stop; $merge++ } catch {}
    }
  }
}
Write-Output ("Pastas renomeadas (sufixo removido): {0} | mescladas: {1}" -f $ren, $merge)
$rem = @(Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match ' \(\d+\)$' }).Count
Write-Output ("Pastas ainda com sufixo (N): {0}" -f $rem)