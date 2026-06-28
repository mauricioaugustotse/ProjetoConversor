# Esvazia o Cofre Pessoal: unicos -> categorias (subpastas pelo mapa, soltos pela IA), duplicatas -> Lixeira.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cofre='C:\Users\mauri\OneDrive\Cofre Pessoal'
# categorias por prefixo
$cats=@{}; foreach($d in (Get-ChildItem -LiteralPath $base -Directory -Force)){ if($d.Name -match '^(\d{2}) '){ $cats[$Matches[1]]=$d.FullName } }
$mapaSub=Get-Content -LiteralPath "$out\mapa_cofre_sub.json" -Encoding UTF8|ConvertFrom-Json
$destChave=Get-Content -LiteralPath "$out\destinos_cofre.json" -Encoding UTF8|ConvertFrom-Json
# dup set
$dupSet=New-Object System.Collections.Generic.HashSet[string]
foreach($l in (Get-Content -LiteralPath "$out\cofre_dup.txt" -Encoding UTF8)){ if($l.Trim()){ $w=($l -replace '^/c/','C:/' -replace '^/C/','C:/'); $w=$w.Replace('/','\'); [void]$dupSet.Add($w.ToLower()) } }
# soltos -> chave
$soltoChave=@{}
$mapS=@{}; foreach($m in (Import-Csv "$out\map_cofre_soltos.csv" -Encoding UTF8)){ $mapS[[int]$m.ID]=$m.Full }
foreach($f in (Get-ChildItem "$out\lotes_cofre_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$' -and $mapS.ContainsKey([int]$r.id)){ $soltoChave[$mapS[[int]$r.id].ToLower()]=$r.chave } } }
function DestDir($spec){ $p=$spec -split '\|',2; $dir=$cats[$p[0]]; if($p.Count -gt 1 -and $p[1]){ $dir=Join-Path $dir $p[1] }; return $dir }

$plan=New-Object System.Collections.Generic.List[object]
# percorrer todo o cofre
foreach($f in (Get-ChildItem -LiteralPath $cofre -Recurse -File -Force -EA SilentlyContinue | Where-Object {$_.Name -ne 'desktop.ini'})){
  if($dupSet.Contains($f.FullName.ToLower())){ $plan.Add([pscustomobject]@{Acao='Lixeira';De=$f.FullName;Para='';Grp='dup'}); continue }
  $rel=$f.FullName.Substring($cofre.Length+1); $parts=$rel -split '\\',2
  if($parts.Count -eq 1){
    # solto na raiz -> por chave
    $ch=$soltoChave[$f.FullName.ToLower()]; if(-not $ch){ $ch='Outros' }
    $dir=DestDir $destChave.$ch
    $plan.Add([pscustomobject]@{Acao='Mover';De=$f.FullName;Para=(Join-Path $dir $f.Name);Grp=("solto:"+$ch)})
  } else {
    # dentro de subpasta de 1o nivel
    $sub1=$parts[0]; $pref=$mapaSub.$sub1
    if(-not $pref){ $plan.Add([pscustomobject]@{Acao='Mover';De=$f.FullName;Para=(Join-Path (DestDir '01|Outros documentos (do Cofre)') $f.Name);Grp='sub:?'}); continue }
    $destBase=Join-Path $cats[$pref] $sub1   # ex: 03-Financeiro\IRPF
    $destPara=Join-Path $destBase ($rel.Substring($sub1.Length+1))
    $plan.Add([pscustomobject]@{Acao='Mover';De=$f.FullName;Para=$destPara;Grp=("sub:"+$sub1)})
  }
}
$plan|Export-Csv "$out\plano_cofre.csv" -NoTypeInformation -Encoding UTF8
$mv=@($plan|Where-Object Acao -eq 'Mover'); $lx=@($plan|Where-Object Acao -eq 'Lixeira')
Write-Output ("Mover: {0} | Lixeira (duplicatas): {1}" -f $mv.Count,$lx.Count)
Write-Output "`n=== Destino por grupo ==="
$mv|Group-Object{ $_.Para.Substring($base.Length+1).Split('\')[0..1] -join '\' }|Sort-Object Count -Descending|ForEach-Object{ Write-Output ("   {0,4}  {1}" -f $_.Count,$_.Name) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_cofre_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $okM=0;$okL=0;$err=0
  foreach($p in $mv){
    try{ $dir=[IO.Path]::GetDirectoryName($p.Para); [IO.Directory]::CreateDirectory($dir)|Out-Null
      $dest=$p.Para; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $dir ("$b ($i)$e"); $i++ } }
      [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{Acao='Mover';De=$p.De;Para=$dest}); $okM++
    }catch{ $err++ }
  }
  foreach($p in $lx){ try{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p.De,'OnlyErrorDialogs','SendToRecycleBin'); $rows.Add([pscustomobject]@{Acao='Lixeira';De=$p.De;Para=''}); $okL++ }catch{ $err++ } }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  # esvaziar cofre (pastas vazias)
  $vaz=Get-ChildItem -LiteralPath $cofre -Recurse -Directory -Force -EA SilentlyContinue|Sort-Object {($_.FullName -split '\\').Count} -Descending
  $remP=0; foreach($v in $vaz){ if(@(Get-ChildItem -LiteralPath $v.FullName -Recurse -File -Force -EA SilentlyContinue|Where-Object{$_.Name -ne 'desktop.ini'}).Count -eq 0){ try{[IO.Directory]::Delete($v.FullName,$true);$remP++}catch{} } }
  Write-Output ("`nMovidos: {0} | Lixeira: {1} | Erros: {2} | Pastas vazias do cofre removidas: {3} | Log: {4}" -f $okM,$okL,$err,$remP,$log)
}