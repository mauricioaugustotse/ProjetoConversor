# Aplica a renomeacao por conteudo (Documentos). Pareia por posicao. Mantem acentos. -Execute aplica.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$map=@{}; foreach($m in (Import-Csv "$out\map_ren.csv" -Encoding UTF8)){ $map[[int]$m.ID]=$m }
# pareamento por ID (os ids do Opus sao globais corretos; robusto a lotes incompletos)
$cls=@{}
foreach($f in (Get-ChildItem "$out\lotes_ren_out\res_*.csv")){
  foreach($r in (Import-Csv -LiteralPath $f.FullName -Delimiter '|' -Encoding UTF8)){
    if($r.id -match '^\d+$'){ $cls[[int]$r.id]=$r }
  }
}
Write-Output ("Map: {0} | classificacoes pareadas: {1}" -f $map.Count, $cls.Count)

function SanitizeName($s){
  if([string]::IsNullOrWhiteSpace($s)){ return '' }
  $s = $s -replace '[\\/:\*\?"<>\|]',' '
  $s = ($s -replace '\s{2,}',' ').Trim().TrimEnd('.',' ')
  return $s
}
$plan=New-Object System.Collections.Generic.List[object]
$used=@{}; $mud=0
foreach($id in ($map.Keys|Sort-Object)){
  if(-not $cls.ContainsKey($id)){ continue }
  $a=$map[$id]; $novo=SanitizeName $cls[$id].nomeNovo
  if([string]::IsNullOrWhiteSpace($novo)){ continue }
  $dir=[IO.Path]::GetDirectoryName($a.Path); $ext=$a.Ext
  # limite caminho
  $maxN=[Math]::Max(12, 255 - $dir.Length - 1 - $ext.Length - 6)
  if($novo.Length -gt $maxN){
    $novo=$novo.Substring(0,$maxN)
    # cortar na ultima fronteira de palavra (se nao perder demais)
    $sp=$novo.LastIndexOf(' ')
    if($sp -gt 0 -and $sp -ge ($maxN-18)){ $novo=$novo.Substring(0,$sp) }
    # remover cauda orfa (separador/preposicao penduradao no fim) - apenas ASCII no .ps1
    $endash=[char]0x2013
    $trimChars=[char[]]@('.',' ','-',$endash)
    $novo=$novo.TrimEnd($trimChars)
    $rxDash='\s*[-'+$endash+']\s*$'; $novo=$novo -replace $rxDash,''
    $ordm=[char]0x00ba; $ordf=[char]0x00b0
    $rxPrep='(?i)\s+(n|n'+$ordm+'|n'+$ordf+'|de|da|do|dos|das|e|-)\s*$'; $novo=$novo -replace $rxPrep,''
    $novo=$novo.TrimEnd($trimChars)
  }
  $newName=$novo+$ext
  if($newName -eq $a.Nome){ continue }   # nome ja igual
  $dest=Join-Path $dir $newName
  $key=$dest.ToLower()
  while($used.ContainsKey($key) -or (Test-Path -LiteralPath $dest)){
    $i=if($used.ContainsKey($key+'#')){ $used[$key+'#']+1 }else{2}; $used[$key+'#']=$i
    $newName="$novo ($i)$ext"; $dest=Join-Path $dir $newName; $key=$dest.ToLower()
    if($i -gt 60){ break }
  }
  $used[$key]=1
  $plan.Add([pscustomobject]@{ De=$a.Path; Para=$dest; Pasta=$a.Pasta; NomeAtual=$a.Nome; NomeNovo=(Split-Path $dest -Leaf) }); $mud++
}
$plan | Export-Csv "$out\plano_ren.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Arquivos que mudam de nome: {0}" -f $mud)
$plan | Select-Object -First 12 | ForEach-Object { Write-Output ("  {0}`n   -> {1}" -f (Split-Path $_.De -Leaf), (Split-Path $_.Para -Leaf)) }
$maxLen=($plan|ForEach-Object{$_.Para.Length}|Measure-Object -Maximum).Maximum
$dups=@($plan|Group-Object{$_.Para.ToLower()}|Where-Object Count -gt 1).Count
Write-Output ("Maior caminho: {0} | destinos duplicados: {1}" -f $maxLen,$dups)

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_ren_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    try{ [IO.File]::Move($p.De,$p.Para); $rows.Add([pscustomobject]@{De=$p.De;Para=$p.Para}); $ok++ }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}