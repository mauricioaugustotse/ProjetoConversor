# Fase A: desfazer o backup ONE DRIVE aninhado em TSE.
# Duplicatas -> Lixeira; .lnk -> Lixeira; TESE unicos -> Estudos\TESE; resto unico -> promovido p/ TSE\<tema>.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$catJur=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '05 *' } | Select-Object -First 1
$catEst=Get-ChildItem -LiteralPath $base -Directory -Force | Where-Object { $_.Name -like '06 *' } | Select-Object -First 1
$tse=Join-Path $catJur.FullName 'TSE'
$onedrive=Join-Path $tse 'ONE DRIVE'
$teseDst=Join-Path (Join-Path $catEst.FullName 'TESE') '(Recuperados do backup OneDrive)'

function ToWin($p){ ($p -replace '^/c/','C:/' -replace '^/C/','C:/').Replace('/','\') }

$plan=New-Object System.Collections.Generic.List[object]
# 1) duplicatas -> Lixeira
foreach($l in (Get-Content -LiteralPath "$out\backup_dup.txt" -Encoding UTF8)){ if($l.Trim()){ $plan.Add([pscustomobject]@{ De=(ToWin $l); Acao='Lixeira'; Para='' }) } }
# 2) unicos
foreach($l in (Get-Content -LiteralPath "$out\backup_uniq.txt" -Encoding UTF8)){
  if(-not $l.Trim()){ continue }
  $w=ToWin $l
  $rel=$w.Substring($onedrive.Length+1)   # subpath dentro de ONE DRIVE
  if($w.ToLower().EndsWith('.lnk')){ $plan.Add([pscustomobject]@{ De=$w; Acao='Lixeira'; Para='' }); continue }
  if($rel -like 'Documentos\TESE\*'){ $sub=$rel.Substring('Documentos\TESE\'.Length); $plan.Add([pscustomobject]@{ De=$w; Acao='Mover'; Para=(Join-Path $teseDst $sub) }); continue }
  $rel2 = if($rel -like 'Documentos\*'){ $rel.Substring('Documentos\'.Length) } else { $rel }
  $plan.Add([pscustomobject]@{ De=$w; Acao='Mover'; Para=(Join-Path $tse $rel2) })
}
$lix=@($plan|Where-Object Acao -eq 'Lixeira').Count
$mov=@($plan|Where-Object Acao -eq 'Mover')
$plan | Export-Csv "$out\plano_faseA.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Para Lixeira: {0} (1184 duplicatas + .lnk) | Mover (promover): {1}" -f $lix,$mov.Count)
Write-Output "`n-- destinos dos movidos, por pasta de 1o nivel em TSE --"
$mov | Group-Object { ($_.Para.Substring($tse.Length+1) -split '\\')[0] } | Sort-Object Count -Descending | ForEach-Object { Write-Output ("   {0,4}  TSE\{1}" -f $_.Count,$_.Name) }
$movTese=@($mov|Where-Object{ $_.Para -like "$teseDst*" }).Count
Write-Output ("   {0,4}  -> Estudos\TESE\(Recuperados do backup OneDrive)" -f $movTese)
# colisoes de destino
$col=@($mov | Where-Object { Test-Path -LiteralPath $_.Para }).Count
Write-Output ("Colisoes de nome no destino: {0}" -f $col)

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_faseA_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $okM=0;$okL=0;$err=0
  # mover primeiro (preserva), depois lixeira
  foreach($p in ($plan|Where-Object Acao -eq 'Mover')){
    try{
      $dir=[IO.Path]::GetDirectoryName($p.Para); [IO.Directory]::CreateDirectory($dir)|Out-Null
      $dest=$p.Para
      if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $dir ("$b ($i)$e"); $i++ } }
      [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{Acao='Mover';De=$p.De;Para=$dest}); $okM++
    }catch{ $err++ }
  }
  foreach($p in ($plan|Where-Object Acao -eq 'Lixeira')){
    try{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p.De,'OnlyErrorDialogs','SendToRecycleBin'); $rows.Add([pscustomobject]@{Acao='Lixeira';De=$p.De;Para=''}); $okL++ }catch{ $err++ }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  # remover casca vazia ONE DRIVE (se sem arquivos)
  $rest=@(Get-ChildItem -LiteralPath $onedrive -Recurse -File -Force -EA SilentlyContinue).Count
  $cascaMsg = if($rest -eq 0){ try{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($onedrive,'OnlyErrorDialogs','SendToRecycleBin'); 'casca ONE DRIVE removida' }catch{ "casca nao removida: $($_.Exception.Message)" } } else { "casca ainda tem $rest arquivos" }
  Write-Output ("`nMovidos: {0} | Lixeira: {1} | Erros: {2}" -f $okM,$okL,$err)
  Write-Output ("Casca: {0}" -f $cascaMsg)
  Write-Output ("Log: {0}" -f $log)
}