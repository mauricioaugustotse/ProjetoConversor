# Consolida subpastas variantes/typos/lixo para os nomes canonicos da taxonomia. -Execute aplica.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\mauri\HD_Mau'

$canon = @{
  juridico = @('Administrativo','Ambiental','Civel','Constitucional','Consumidor','Eleitoral','Empresarial','Familia e Sucessoes','Outros','Penal','Previdenciario','Trabalhista','Tributario')
  estudos  = @('Direito Administrativo','Direito Ambiental','Direito Civil','Direito Constitucional','Direito do Trabalho','Direito Eleitoral','Direito Empresarial','Direito Internacional','Direito Penal','Direito Previdenciario','Direito Processual Civil','Direito Processual Penal','Direito Tributario','Portugues','Raciocinio Logico e Matematica','Informatica','Atualidades e Outros')
  admin    = @('Atas e sessoes','Comunicacoes internas','Contratos e termos','Editais','Formularios','Listas e planilhas','Oficios e memorandos','Outros','Portarias e resolucoes')
  pessoal  = @('Comprovantes e recibos','Declaracoes','Documentos pessoais','Faturas e boletos','Impostos','Outros')
}
$syn = @{
  juridico = @{ 'Civil'='Civel'; 'Comercial'='Empresarial'; 'Trabalho'='Trabalhista'; 'Processual'='Outros'; 'Educacao'='Outros' }
  estudos  = @{ 'Lingua Portuguesa'='Portugues'; 'Direito Consumedor'='Atualidades e Outros'; 'Direito Consumidor'='Atualidades e Outros'; 'Direito Educacao'='Atualidades e Outros'; 'Alemao'='Atualidades e Outros'; 'Familia e Sucessoes'='Direito Civil'; 'Outros'='Atualidades e Outros' }
  admin    = @{ 'Administrativo'='Outros'; 'Circulares e memorandos'='Oficios e memorandos'; 'Relatorios e atas'='Atas e sessoes' }
  pessoal  = @{ 'Formularios'='Outros'; 'Listas e planilhas'='Outros'; 'Extratos e movimentacao'='Outros' }
}
$default = @{ juridico='Outros'; estudos='Atualidades e Outros'; admin='Outros'; pessoal='Outros' }

$plan = New-Object System.Collections.Generic.List[object]
function AddMove($srcDir, $dstDir){
  $n = (Get-ChildItem -LiteralPath $srcDir -Recurse -File -Force -EA SilentlyContinue | Measure-Object).Count
  $plan.Add([pscustomobject]@{ Src=$srcDir; Dst=$dstDir; N=$n })
}

# 01 - Juridico: Tipo / Materia
$jur = Join-Path $base '01 - Juridico'
foreach($tipo in (Get-ChildItem -LiteralPath $jur -Directory -Force)){
  foreach($mat in (Get-ChildItem -LiteralPath $tipo.FullName -Directory -Force)){
    if($canon.juridico -contains $mat.Name){ continue }
    $d = if($syn.juridico.ContainsKey($mat.Name)){ $syn.juridico[$mat.Name] } else { $default.juridico }
    AddMove $mat.FullName (Join-Path $tipo.FullName $d)
  }
}
# 02/03/04: subpasta direta
$map = @{ '02 - Estudos e concursos'='estudos'; '03 - Administrativo'='admin'; '04 - Pessoal e financeiro'='pessoal' }
foreach($catName in $map.Keys){
  $key = $map[$catName]; $catDir = Join-Path $base $catName
  foreach($sub in (Get-ChildItem -LiteralPath $catDir -Directory -Force)){
    if($canon[$key] -contains $sub.Name){ continue }
    # caso especial: Documentos pessoais em Administrativo -> Pessoal
    if($key -eq 'admin' -and $sub.Name -eq 'Documentos pessoais'){ AddMove $sub.FullName (Join-Path $base '04 - Pessoal e financeiro\Documentos pessoais'); continue }
    $d = if($syn[$key].ContainsKey($sub.Name)){ $syn[$key][$sub.Name] } else { $default[$key] }
    AddMove $sub.FullName (Join-Path $catDir $d)
  }
}

Write-Output ("Pastas a consolidar: {0} | arquivos afetados: {1}" -f $plan.Count, ($plan|Measure-Object N -Sum).Sum)
$plan | Sort-Object { (Split-Path (Split-Path $_.Src -Parent) -Leaf) } | ForEach-Object {
  Write-Output ("  [{0,4}]  ...\{1}\{2}  ->  {3}" -f $_.N, (Split-Path (Split-Path $_.Src -Parent) -Leaf), (Split-Path $_.Src -Leaf), (Split-Path $_.Dst -Leaf))
}

if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="C:\Users\mauri\ProjetoConversor\log_consolida_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $moved=0; $err=0
  foreach($p in $plan){
    if(-not (Test-Path -LiteralPath $p.Dst)){ [IO.Directory]::CreateDirectory($p.Dst)|Out-Null }
    foreach($f in (Get-ChildItem -LiteralPath $p.Src -Recurse -File -Force -EA SilentlyContinue)){
      $tgt = Join-Path $p.Dst $f.Name
      if(Test-Path -LiteralPath $tgt){ $bn=[IO.Path]::GetFileNameWithoutExtension($f.Name);$ext=$f.Extension;$i=2; do{ $tgt=Join-Path $p.Dst "$bn ($i)$ext";$i++ }while(Test-Path -LiteralPath $tgt) }
      try { [IO.File]::Move($f.FullName,$tgt); $rows.Add([pscustomobject]@{De=$f.FullName;Para=$tgt}); $moved++ } catch { $err++ }
    }
    # remover pasta variante se vazia
    if((Get-ChildItem -LiteralPath $p.Src -Recurse -Force -EA SilentlyContinue|Measure-Object).Count -eq 0){ Remove-Item -LiteralPath $p.Src -Recurse -Force -EA SilentlyContinue }
  }
  $rows | Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nMovidos: {0} | Erros: {1} | Log: {2}" -f $moved,$err,$log)
}