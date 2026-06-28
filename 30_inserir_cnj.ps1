# Insere o numero de processo CNJ no nome dos arquivos que o tem no conteudo mas nao no nome. -Execute aplica.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\mauri\HD_Mau'; $out='C:\Users\mauri\ProjetoConversor'
$meta = Import-Csv "$out\conteudo_metadados.csv" -Encoding UTF8 | Where-Object { $_.CNJ -ne '' }

function FmtCnj($c){
  $d = $c -replace '\D',''
  if($d.Length -eq 20){ return ('{0}-{1}.{2}.{3}.{4}.{5}' -f $d.Substring(0,7),$d.Substring(7,2),$d.Substring(9,4),$d.Substring(13,1),$d.Substring(14,2),$d.Substring(16,4)) }
  return ($c -replace '[^0-9.\-]','')
}

$plan = New-Object System.Collections.Generic.List[object]
foreach($m in $meta){
  $full = Join-Path $base $m.Rel
  if(-not (Test-Path -LiteralPath $full)){ continue }
  $numCnj = ($m.CNJ -replace '\D','')
  if($numCnj.Length -ne 20){ continue }
  $nome = [IO.Path]::GetFileName($full)
  if(($nome -replace '\D','') -like "*$numCnj*"){ continue }   # ja tem
  $b = [IO.Path]::GetFileNameWithoutExtension($nome); $ext=[IO.Path]::GetExtension($nome)
  $cnjF = FmtCnj $m.CNJ
  $dir = [IO.Path]::GetDirectoryName($full)
  # limite de caminho: truncar base se necessario
  $suf = " - CNJ $cnjF"
  $maxBase = [Math]::Max(10, 255 - $dir.Length - 1 - $suf.Length - $ext.Length)
  if($b.Length -gt $maxBase){ $b = $b.Substring(0,$maxBase).TrimEnd('.',' ') }
  $novo = "$b$suf$ext"
  $dest = Join-Path $dir $novo
  if(Test-Path -LiteralPath $dest){ continue }
  $plan.Add([pscustomobject]@{ De=$full; Novo=$novo; Dest=$dest })
}
Write-Output ("Candidatos a receber CNJ no nome: {0}" -f $plan.Count)
Write-Output "`n=== Amostra (8) ==="
$plan | Select-Object -First 8 | ForEach-Object { Write-Output ("  {0}`n   -> {1}" -f (Split-Path $_.De -Leaf), $_.Novo) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_cnj_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try { Rename-Item -LiteralPath $p.De -NewName $p.Novo -ErrorAction Stop; $rows.Add([pscustomobject]@{De=$p.De;Para=$p.Dest}); $ok++ }
    catch { $err++ }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}