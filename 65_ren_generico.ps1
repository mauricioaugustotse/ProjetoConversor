# Renomeia arquivos de uma subpasta de STs a partir de map(id<TAB>nome) + res(id|nomeNovo). Sem hardcode de acentos.
param([Parameter(Mandatory=$true)][string]$SubName, [Parameter(Mandatory=$true)][string]$MapTsv, [Parameter(Mandatory=$true)][string]$ResCsv, [switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$c12=Get-ChildItem -LiteralPath $base -Directory -Force|Where-Object{$_.Name -like '12 *'}|Select-Object -First 1
$sts=Get-ChildItem -LiteralPath $c12.FullName -Directory -Force|Where-Object{$_.Name -ceq 'STs'}|Select-Object -First 1
$sub=Get-ChildItem -LiteralPath $sts.FullName -Directory -Force|Where-Object{$_.Name -ceq $SubName}|Select-Object -First 1
if(-not $sub){ Write-Output "Subpasta nao encontrada: $SubName"; return }
$d=$sub.FullName
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$map=@{}; foreach($l in (Get-Content -LiteralPath $MapTsv -Encoding UTF8)){ $p=$l -split "`t",2; if($p.Count -eq 2){ $map[[int]$p[0]]=$p[1] } }
$cls=@{}; foreach($r in (Import-Csv -LiteralPath $ResCsv -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.nomeNovo } }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $atual=$map[$id]
  if(-not (Test-Path -LiteralPath (Join-Path $d $atual))){ Write-Output ("   [FALTA - movido?] {0}" -f $atual); continue }
  $ext=[IO.Path]::GetExtension($atual); $novo=(Sanitize $cls[$id])+$ext
  if($novo -eq $atual){ continue }
  $plan+=[pscustomobject]@{ De=(Join-Path $d $atual); Para=(Join-Path $d $novo); Atual=$atual; Novo=$novo }
}
Write-Output ("[$SubName] a renomear: {0} de {1}" -f $plan.Count,$map.Count)
$plan|ForEach-Object{ Write-Output ("   {0}`n      -> {1}" -f $_.Atual,$_.Novo) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_ren_${SubName}_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ $dest=$p.Para; if((Test-Path -LiteralPath $dest) -and ($dest -ne $p.De)){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $d ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}