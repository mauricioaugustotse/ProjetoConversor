# Renomeia fotos de nome fraco -> "<contexto da pasta> - <data> [(k)]". Mantem acentos (vem do filesystem).
# -Execute aplica. Sem ele, dry-run + plano_fotos.csv.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'
$excl="$base\LUIZ CELSO VIEIRA"
Add-Type -AssemblyName System.Drawing

function ExifDate($path){
  try{
    $img=[System.Drawing.Image]::FromFile($path)
    try{
      $p=$img.GetPropertyItem(36867)  # DateTimeOriginal "yyyy:MM:dd HH:mm:ss"
      $s=[Text.Encoding]::ASCII.GetString($p.Value)
      if($s.Length -ge 10){ return ($s.Substring(0,10) -replace ':','-') }
    } catch {} finally { $img.Dispose() }
  } catch {}
  return $null
}
function Fraco($b){
  $l=$b.ToLower()
  if($l -match '^(img|img_e|dsc|dscn|dscf|dcim|pxl|pano|photo|foto|scan|scn|screenshot|screen shot|captura de tela|capturadetela|win_|wp_|vid|mvimg|burst|inshot|snapchat|fb_img|fb img|image|images|download|unnamed|received|polish|layout|collage|gopro|gp__|hpim|cimg|sam_|sdc1|p_?\d|20[0-2]\d[01]\d[0-3]\d|19\d{6})'){ return $true }
  if(($b -replace '[^a-zA-Z]','').Length -le 2 -and $b.Length -ge 4){ return $true }
  if($b -match '^[\d _.\-()]+$'){ return $true }
  return $false
}
function Contexto($dir){
  $name=Split-Path $dir -Leaf
  if($name.ToLower() -match '^(fotos?|imagens?|img|images?|pictures?|fotografias?|digitalizad[oa]s?|scans?|galeria|camera|dcim|pics?)$'){
    $parent=Split-Path (Split-Path $dir -Parent) -Leaf
    if($parent){ $name="$parent - $name" }
  }
  return ($name -replace '^\d{2} - ','')
}
function DataFoto($f){
  $n=[IO.Path]::GetFileNameWithoutExtension($f.Name)
  if($n -match '(20[0-2]\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])'){ return ("{0}-{1}-{2}" -f $matches[1],$matches[2],$matches[3]) }
  $e=$f.Extension.ToLower()
  if($e -eq '.jpg' -or $e -eq '.jpeg'){ $d=ExifDate $f.FullName; if($d){ return $d } }
  $dt=if($f.CreationTime -lt $f.LastWriteTime){$f.CreationTime}else{$f.LastWriteTime}
  return $dt.ToString('yyyy-MM-dd')
}

$exts=@('.jpg','.jpeg','.png')
$fotos = Get-ChildItem -LiteralPath $base -Recurse -File -Force -EA SilentlyContinue | Where-Object {
  $exts -contains $_.Extension.ToLower() -and $_.FullName -notlike "$excl*" -and (Fraco ([IO.Path]::GetFileNameWithoutExtension($_.Name)))
}
Write-Output ("Fotos de nome fraco encontradas: {0}" -f @($fotos).Count)

# montar entradas com grupo
$ent=foreach($f in $fotos){
  $dir=$f.DirectoryName; $ctx=Contexto $dir; $data=DataFoto $f
  [pscustomobject]@{ File=$f; Dir=$dir; Ctx=$ctx; Data=$data; Key=("{0}|{1}|{2}" -f $dir,$ctx,$data); OrdName=$f.Name }
}
# numerar por grupo
$plan=New-Object System.Collections.Generic.List[object]
foreach($g in ($ent | Group-Object Key)){
  $items=@($g.Group | Sort-Object OrdName)
  $multi=$items.Count -gt 1; $k=0
  foreach($it in $items){
    $k++
    $san = ($it.Ctx -replace '[\\/:\*\?"<>\|]',' '); $san=($san -replace '\s{2,}',' ').Trim()
    $core = if($multi){ "{0} - {1} ({2})" -f $san,$it.Data,$k } else { "{0} - {1}" -f $san,$it.Data }
    $ext=$it.File.Extension.ToLower()
    $maxN=[Math]::Max(12,255-$it.Dir.Length-1-$ext.Length-6)
    if($core.Length -gt $maxN){ $core=$core.Substring(0,$maxN).TrimEnd('.',' ','-') }
    $newName=$core+$ext
    if($newName -ne $it.File.Name){ $plan.Add([pscustomobject]@{ De=$it.File.FullName; Dir=$it.Dir; Core=$core; Ext=$ext; NomeAtual=$it.File.Name }) }
  }
}
# resolver colisoes (na mesma pasta e com existentes)
$used=@{}; $final=New-Object System.Collections.Generic.List[object]
foreach($p in $plan){
  $name="$($p.Core)$($p.Ext)"; $dest=Join-Path $p.Dir $name; $key=$dest.ToLower(); $i=1
  while($used.ContainsKey($key) -or (Test-Path -LiteralPath $dest)){
    $i++; $name="$($p.Core) ($i)$($p.Ext)"; $dest=Join-Path $p.Dir $name; $key=$dest.ToLower(); if($i -gt 200){break}
  }
  $used[$key]=1; $final.Add([pscustomobject]@{ De=$p.De; Para=$dest; NomeAtual=$p.NomeAtual; NomeNovo=$name })
}
$final | Export-Csv "$out\plano_fotos.csv" -NoTypeInformation -Encoding UTF8
Write-Output ("Fotos a renomear: {0}" -f $final.Count)
$final | Select-Object -First 14 | ForEach-Object { Write-Output ("  {0}`n   -> {1}" -f $_.NomeAtual,$_.NomeNovo) }

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_fotos_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $final){ try{ [IO.File]::Move($p.De,$p.Para); $rows.Add([pscustomobject]@{De=$p.De;Para=$p.Para}); $ok++ }catch{ $err++ } }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeadas: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}