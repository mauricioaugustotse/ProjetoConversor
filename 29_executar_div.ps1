# Executa plano_div.csv (move Diversos classificados por conteudo p/ categorias reais). Log de reversao.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\mauri\HD_Mau'
$out  = 'C:\Users\mauri\ProjetoConversor'
$plan = Import-Csv "$out\plano_div.csv" -Encoding UTF8
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'; $log = "$out\log_div_$ts.csv"
$logRows = New-Object System.Collections.Generic.List[object]
$ok=0;$err=0;$skip=0;$errs=New-Object System.Collections.Generic.List[string]
foreach($r in $plan){
  try {
    if(-not (Test-Path -LiteralPath $r.Origem)){ $skip++; continue }
    $dir=[IO.Path]::GetDirectoryName($r.Destino)
    if(-not $DryRun){ [IO.Directory]::CreateDirectory($dir)|Out-Null; [IO.File]::Move($r.Origem,$r.Destino) }
    $logRows.Add([pscustomobject]@{De=$r.Origem;Para=$r.Destino}); $ok++
  } catch { $err++; if($errs.Count -lt 30){ $errs.Add("$($r.Origem) -> $($_.Exception.Message)") } }
}
$logRows | Export-Csv $log -NoTypeInformation -Encoding UTF8
# remover subpastas vazias em 05-Diversos
if(-not $DryRun){
  Get-ChildItem -LiteralPath (Join-Path $base '05 - Diversos (a revisar)') -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { if((Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue|Measure-Object).Count -eq 0){ Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } }
}
Write-Output ("Movidos: {0} | Pulados: {1} | Erros: {2}" -f $ok,$skip,$err)
$errs | ForEach-Object { Write-Output "  ERRO: $_" }
Write-Output ("Log: {0}" -f $log)