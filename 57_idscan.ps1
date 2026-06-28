# Move documentos de identidade reais do Mauricio (classificados pela IA) para 01\<tipo>, com dedup por hash.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$cats=@{}; foreach($d in (Get-ChildItem -LiteralPath $base -Directory -Force)){ if($d.Name -match '^(\d{2}) '){ $cats[$Matches[1]]=$d.FullName } }
$destChave=Get-Content -LiteralPath "$out\destinos_cofre.json" -Encoding UTF8|ConvertFrom-Json
$tipos='Identidade','CPF','Eleitoral','Militar','Certidoes','Trabalho'
$map=@{}; foreach($m in (Import-Csv "$out\map_idscan.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($f in (Get-ChildItem "$out\lotes_idscan_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.chave } } }
function DestDir($spec){ $p=$spec -split '\|',2; $dir=$cats[$p[0]]; if($p.Count -gt 1 -and $p[1]){ $dir=Join-Path $dir $p[1] }; return $dir }
# cache de hashes por pasta destino
$hashCache=@{}
function HashesDe($dir){ if(-not $hashCache.ContainsKey($dir)){ $h=@{}; if(Test-Path -LiteralPath $dir){ foreach($x in (Get-ChildItem -LiteralPath $dir -File -Force -EA SilentlyContinue)){ try{ $h[(Get-FileHash -LiteralPath $x.FullName -Algorithm MD5).Hash]=$x.Name }catch{} } }; $hashCache[$dir]=$h }; return $hashCache[$dir] }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  $ch=$cls[$id]; if(-not $ch -or ($tipos -notcontains $ch)){ continue }
  $destDir=DestDir $destChave.$ch
  $h=try{(Get-FileHash -LiteralPath $map[$id].Full -Algorithm MD5).Hash}catch{''}
  $dup=$false; if($h){ $dup=(HashesDe $destDir).ContainsKey($h) }
  $acao=if($dup){'Lixeira(dup)'}else{'Mover'}
  $plan+=[pscustomobject]@{ De=$map[$id].Full; DestDir=$destDir; Nome=$map[$id].Nome; Acao=$acao; Tipo=$ch }
}
Write-Output ("Documentos reais a tratar: {0} | Mover: {1} | Duplicatas->Lixeira: {2} | Ficam: {3}" -f $plan.Count, @($plan|Where-Object Acao -eq 'Mover').Count, @($plan|Where-Object Acao -like 'Lixeira*').Count, @($cls.Values|Where-Object{$tipos -notcontains $_}).Count)
$plan|ForEach-Object{ Write-Output ("   [{0}|{1}] {2}  ({3})" -f $_.Tipo,$_.Acao,$_.Nome,(Split-Path $_.De -Parent | Split-Path -Leaf)) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_idscan_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$lx=0;$err=0
  foreach($p in $plan){
    try{
      if($p.Acao -like 'Lixeira*'){ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p.De,'OnlyErrorDialogs','SendToRecycleBin'); $rows.Add([pscustomobject]@{Acao='Lixeira';De=$p.De;Para=''}); $lx++ }
      else { [IO.Directory]::CreateDirectory($p.DestDir)|Out-Null; $dest=Join-Path $p.DestDir $p.Nome; if(Test-Path -LiteralPath $dest){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $p.DestDir ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{Acao='Mover';De=$p.De;Para=$dest}); $ok++ }
    }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Lixeira: {1} | Erros: {2} | Log: {3}" -f $ok,$lx,$err,$log)
}