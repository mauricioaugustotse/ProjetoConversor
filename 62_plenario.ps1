# Fase 4a: separa Plenário em Modelos e pareceres (.docx) e Vídeos de sessões (.mp4); RICD fica na raiz.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$c12=Get-ChildItem -LiteralPath $base -Directory -Force|Where-Object{$_.Name -like '12 *'}|Select-Object -First 1
$pl=Get-ChildItem -LiteralPath $c12.FullName -Directory -Force|Where-Object{$_.Name -like 'Plen*'}|Select-Object -First 1
$plPath=$pl.FullName
$j=Get-Content -LiteralPath "$out\plenario.json" -Encoding UTF8|ConvertFrom-Json
$plan=@()
foreach($f in (Get-ChildItem -LiteralPath $plPath -File -Force|Where-Object{$_.Name -ne 'desktop.ini'})){
  $e=$f.Extension.ToLower(); $sub=$null
  if($e -eq '.mp4'){ $sub=$j.video }
  elseif($e -eq '.docx' -and $f.Name -notmatch 'RICD'){ $sub=$j.docx }
  if($sub){ $plan+=[pscustomobject]@{ De=$f.FullName; DestDir=(Join-Path $plPath $sub); Nome=$f.Name; Sub=$sub } }
}
Write-Output ("Plenário - a mover: {0} (RICD e outros ficam na raiz)" -f $plan.Count)
$plan|Group-Object Sub|ForEach-Object{ Write-Output ("   {0,2}  {1}" -f $_.Count,$_.Name) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_plenario_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){ try{ [IO.Directory]::CreateDirectory($p.DestDir)|Out-Null; $dest=Join-Path $p.DestDir $p.Nome; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e2=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $p.DestDir ("$b ($i)$e2"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ } }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}