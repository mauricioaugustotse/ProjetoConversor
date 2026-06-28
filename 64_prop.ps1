# Padroniza nomes de STs\Proposições. Pareia por ID. Sem hardcode de acentos.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$c12=Get-ChildItem -LiteralPath $base -Directory -Force|Where-Object{$_.Name -like '12 *'}|Select-Object -First 1
$sts=Get-ChildItem -LiteralPath $c12.FullName -Directory -Force|Where-Object{$_.Name -ceq 'STs'}|Select-Object -First 1
$prop=Get-ChildItem -LiteralPath $sts.FullName -Directory -Force|Where-Object{$_.Name -like 'Propos*'}|Select-Object -First 1
$d=$prop.FullName
function Sanitize($s){ ($s -replace '[\\/:\*\?"<>\|]',' ' -replace '\s{2,}',' ').Trim().TrimEnd('.',' ') }
$map=@{}; foreach($l in (Get-Content -LiteralPath "$out\map_prop.tsv" -Encoding UTF8)){ $p=$l -split "`t",2; if($p.Count -eq 2){ $map[[int]$p[0]]=$p[1] } }
$cls=@{}; foreach($r in (Import-Csv "$out\lotes_prop_out\res_00.csv" -Delimiter '|' -Encoding UTF8)){ if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r.nomeNovo } }
$plan=@()
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $nn=$cls[$id] -replace 'Minuta de\s*-\s*','Minuta de '
  if($nn -match '^\s*Parecer'){ continue }   # pareceres serao movidos/padronizados em STs\Pareceres
  $atual=$map[$id]; $ext=[IO.Path]::GetExtension($atual)
  $novo=(Sanitize $nn)+$ext; if($novo -eq $atual){ continue }
  $plan+=[pscustomobject]@{ De=(Join-Path $d $atual); Para=(Join-Path $d $novo); Atual=$atual; Novo=$novo }
}
Write-Output ("A renomear: {0} de {1}" -f $plan.Count,$map.Count)
$plan|ForEach-Object{ Write-Output ("   {0}`n      -> {1}" -f $_.Atual,$_.Novo) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_prop_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ $dest=$p.Para; if((Test-Path -LiteralPath $dest) -and ($dest -ne $p.De)){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $d ("$b ($i)$e"); $i++ } }; [IO.File]::Move($p.De,$dest); $rows.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++ }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}