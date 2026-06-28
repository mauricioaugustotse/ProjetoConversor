# Move os arquivos da pasta ARQUIVOS para as categorias conforme arquivos_class.csv. -Execute aplica.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$arq="$base\11 - Diversos (a revisar)\ARQUIVOS"
$cls = Import-Csv "$out\arquivos_class.csv" -Delimiter '|' -Encoding UTF8
$catsOk = @('01 - Pessoal e identidade','02 - Saúde','03 - Financeiro','04 - Imóveis','05 - Jurídico e trabalho','06 - Estudos, concursos e leitura','07 - Viagens','08 - Veículos','09 - Família, religião e mensagens','10 - Tecnologia e dispositivos','11 - Diversos (a revisar)')
$plan=New-Object System.Collections.Generic.List[object]
foreach($c in $cls){
  $src=Join-Path $arq $c.arquivo
  if(-not (Test-Path -LiteralPath $src)){ continue }
  $cat = $c.categoria; if($catsOk -notcontains $cat){ $cat='11 - Diversos (a revisar)' }
  $plan.Add([pscustomobject]@{ Src=$src; Cat=$cat })
}
Write-Output ("Arquivos a mover: {0}" -f $plan.Count)
$plan | Group-Object Cat | Sort-Object Name | ForEach-Object { Write-Output ("  {0,3}  {1}" -f $_.Count, $_.Name) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_doc_arquivos_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    $catDir=Join-Path $base $p.Cat
    if(-not (Test-Path -LiteralPath $catDir)){ [IO.Directory]::CreateDirectory($catDir)|Out-Null }
    $name=[IO.Path]::GetFileName($p.Src); $dst=Join-Path $catDir $name
    if(Test-Path -LiteralPath $dst){ $bn=[IO.Path]::GetFileNameWithoutExtension($name);$ext=[IO.Path]::GetExtension($name);$i=2; do{ $dst=Join-Path $catDir "$bn ($i)$ext";$i++ }while(Test-Path -LiteralPath $dst) }
    try { [IO.File]::Move($p.Src,$dst); $rows.Add([pscustomobject]@{De=$p.Src;Para=$dst}); $ok++ } catch { $err++ }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  # remover ARQUIVOS se vazia
  if(Test-Path -LiteralPath $arq){ if((Get-ChildItem -LiteralPath $arq -Recurse -Force|Measure-Object).Count -eq 0){ Remove-Item -LiteralPath $arq -Recurse -Force; Write-Output "ARQUIVOS removida (vazia)." } else { Write-Output ("ARQUIVOS ainda tem {0} item(ns)." -f (Get-ChildItem -LiteralPath $arq -Recurse -File -Force|Measure-Object).Count) } }
  Write-Output ("Movidos: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}