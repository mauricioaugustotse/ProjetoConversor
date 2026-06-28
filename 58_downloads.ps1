# Renomeia e move arquivos de Downloads para Documentos conforme classificacao da IA. Sem hardcode de acentos.
param([switch]$Execute)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
function ToAscii($s){ $r=$s.Normalize([Text.NormalizationForm]::FormD); -join ($r.ToCharArray()|Where-Object{[Globalization.CharUnicodeInfo]::GetUnicodeCategory($_) -ne 'NonSpacingMark'}) }
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$catDirs=Get-ChildItem -LiteralPath $base -Directory -Force
function ResolveDest([string]$destRel){
  $parts=$destRel -split '\\',2; $pref=($parts[0].Trim() -split ' ')[0]
  $cat=$catDirs|Where-Object{ ($_.Name -split ' ')[0] -eq $pref }|Select-Object -First 1
  if(-not $cat){ return $null }
  $dir=$cat.FullName
  if($parts.Count -gt 1 -and $parts[1].Trim()){
    $sname=$parts[1].Trim()
    $sub=Get-ChildItem -LiteralPath $dir -Directory -Force -EA SilentlyContinue|Where-Object{ $_.Name -eq $sname }|Select-Object -First 1
    if(-not $sub){ $sub=Get-ChildItem -LiteralPath $dir -Directory -Force -EA SilentlyContinue|Where-Object{ (ToAscii $_.Name).ToLower() -eq (ToAscii $sname).ToLower() }|Select-Object -First 1 }
    if($sub){ $dir=$sub.FullName } else { $dir=Join-Path $dir (Sanitize $sname) }
  }
  return $dir
}
$map=@{}; foreach($m in (Import-Csv "$out\map_down.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
$cls=@{}; foreach($f in (Get-ChildItem "$out\lotes_down_out\res_*.csv")){ foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r } } }
$hashCache=@{}
function HashesDe($dir){ if(-not $hashCache.ContainsKey($dir)){ $h=@{}; if(Test-Path -LiteralPath $dir){ foreach($x in (Get-ChildItem -LiteralPath $dir -File -Force -EA SilentlyContinue)){ try{ $h[(Get-FileHash -LiteralPath $x.FullName -Algorithm MD5).Hash]=$x.Name }catch{} } }; $hashCache[$dir]=$h }; return $hashCache[$dir] }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $destDir=ResolveDest $cls[$id].destino; if(-not $destDir){ $destDir=Join-Path $base '11 - Diversos (a revisar)' }
  $novo=Sanitize $cls[$id].nomeNovo; if(-not $novo){ $novo=[IO.Path]::GetFileNameWithoutExtension($map[$id].Nome) }
  $nome="$novo$($map[$id].Ext)"
  $h=try{(Get-FileHash -LiteralPath $map[$id].Full -Algorithm MD5).Hash}catch{''}
  $dup=$false; if($h){ $dup=(HashesDe $destDir).ContainsKey($h) }
  $acao=if($dup){'Lixeira(dup)'}else{'Mover'}
  $existe=if(Test-Path -LiteralPath $destDir){'ok'}else{'NOVA'}
  $plan+=[pscustomobject]@{ De=$map[$id].Full; DestDir=$destDir; Nome=$nome; Acao=$acao; DestExiste=$existe; Rel=$cls[$id].destino }
}
$plan|Select-Object De,DestDir,Nome,Acao,Rel|Export-Csv "$out\plano_down.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Total: {0} | Mover: {1} | Duplicatas->Lixeira: {2}" -f $plan.Count,@($plan|Where-Object Acao -eq 'Mover').Count,@($plan|Where-Object Acao -like 'Lixeira*').Count)
Write-Output "`n=== Destino (pastas NOVAS marcadas) ==="
$plan|Group-Object{ $_.DestDir.Substring($base.Length+1) }|Sort-Object Name|ForEach-Object{ $ex=if(Test-Path -LiteralPath (Join-Path $base $_.Name)){''}else{'  <-- NOVA'}; Write-Output ("   {0,3}  {1}{2}" -f $_.Count,$_.Name,$ex) }
Write-Output "`n=== Amostra (nome novo) ==="
$plan|Select-Object -First 18|ForEach-Object{ Write-Output ("   {0}`n      -> {1}\{2}" -f (Split-Path $_.De -Leaf),$_.DestDir.Substring($base.Length+1),$_.Nome) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_down_$ts.csv"
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