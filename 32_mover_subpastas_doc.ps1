# Move as subpastas tematicas de Documentos para as categorias de alto nivel. -Execute aplica.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\mauri\OneDrive\Documentos'
$out  = 'C:\Users\mauri\ProjetoConversor'
$map = Get-Content -Raw -Encoding UTF8 "$out\doc_mapa_subpastas.json" | ConvertFrom-Json

# validacao
$faltam = @(); foreach($m in $map){ if(-not (Test-Path -LiteralPath (Join-Path $base $m.sub))){ $faltam += $m.sub } }
Write-Output ("Subpastas no mapa: {0} | nao encontradas: {1}" -f $map.Count, $faltam.Count)
$faltam | ForEach-Object { Write-Output "  FALTA: $_" }
Write-Output ("Categorias: {0}" -f (($map | Select-Object -ExpandProperty cat -Unique | Sort-Object) -join ' | '))

if($Execute -and $faltam.Count -eq 0){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_doc_subpastas_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0;$errs=New-Object System.Collections.Generic.List[string]
  foreach($m in $map){
    $src = Join-Path $base $m.sub
    $catDir = Join-Path $base $m.cat
    if(-not (Test-Path -LiteralPath $catDir)){ [IO.Directory]::CreateDirectory($catDir)|Out-Null }
    $dst = Join-Path $catDir $m.sub
    try { [IO.Directory]::Move($src, $dst); $rows.Add([pscustomobject]@{De=$src;Para=$dst}); $ok++ }
    catch { $err++; if($errs.Count -lt 20){ $errs.Add("$($m.sub) -> $($_.Exception.Message)") } }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidas: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
  $errs | ForEach-Object { Write-Output "  ERRO: $_" }
}