# Renomeia arquivos numa pasta a partir de um CSV (De|Para, UTF-8). Reutilizavel p/ padronizacao.
param([Parameter(Mandatory=$true)][string]$Pasta, [Parameter(Mandatory=$true)][string]$Csv, [switch]$Execute)
$ErrorActionPreference='Stop'
$out='C:\Users\mauri\ProjetoConversor'
$rows=Import-Csv -LiteralPath $Csv -Delimiter '|' -Encoding UTF8
Write-Output ("Pasta: {0}" -f $Pasta)
$plan=@()
foreach($r in $rows){
  $de=Join-Path $Pasta $r.De
  if(-not (Test-Path -LiteralPath $de)){ Write-Output ("   [FALTA] {0}" -f $r.De); continue }
  $para=Join-Path $Pasta $r.Para
  $plan+=[pscustomobject]@{De=$de;Para=$para;NomeDe=$r.De;NomePara=$r.Para}
  Write-Output ("   {0}`n     -> {1}" -f $r.De,$r.Para)
}
Write-Output ("`nA renomear: {0}" -f $plan.Count)
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_padroniza_$ts.csv"
  $rl=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{
      $dest=$p.Para
      if((Test-Path -LiteralPath $dest) -and ($dest -ne $p.De)){ $b=[IO.Path]::GetFileNameWithoutExtension($dest); $e=[IO.Path]::GetExtension($dest); $d=[IO.Path]::GetDirectoryName($dest); $i=2; while(Test-Path -LiteralPath $dest){ $dest=Join-Path $d ("$b ($i)$e"); $i++ } }
      [IO.File]::Move($p.De,$dest); $rl.Add([pscustomobject]@{De=$p.De;Para=$dest}); $ok++
    }catch{ $err++; Write-Output ("   ERRO: {0}" -f $_.Exception.Message) }
  }
  $rl|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}