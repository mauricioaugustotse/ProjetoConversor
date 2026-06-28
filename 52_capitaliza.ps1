# Padroniza capitalizacao de folders -> Title Case (preserva siglas e conectivos).
# Pega folders com qualquer palavra em MAIUSCULAS (3+ letras) que nao seja sigla.
param([switch]$Execute)
$ErrorActionPreference='Stop'
$base='C:\Users\mauri\OneDrive\Documentos'; $out='C:\Users\mauri\ProjetoConversor'; $excl="$base\LUIZ CELSO VIEIRA"
$siglas=('OAB CPF CTPS TSE CD DOU IRPF CNIS ASO PCMSO SEI RG CNH MS PEC PL PLOA LOA FGV TRF1 TRF6 TRF UFMG CLT PIS PASEP AGEL CONLE STF STJ TST TCU CEF BRB FHE POUPEX HRV CRLV PID CMO CCJC PRC PDL DVS IPTU TLP RTD RGI ITBI CEI CRM UH SQS SQSW SQNW UNB ESPCEX ENAM ENFAM MPF MPDFT MPE MPMG MPGO MPSP MPRJ PGE PGDF PGFN BNDES IGESDF SGP CAAPLE SEDID SEDEG SEGJUR ASSEC STI DAA UDN PA AC RE ADC ADI ADPF GRU GPS NFE NFCE TI II III IV ICMS COFINS CSLL IRPJ ISS LGPD CNPJ CEP INSS FGTS IPVA SUS RFB DIRF DARF UF EUA DF MG SP RJ BA GO MT PR SC RS PE CE PA AM RO MA PB RN PI AL SE ES TO AP RR DEFIN CLN CLS SQN GAN SIAFI CAUC CADIN IPOL RCCN COJULEG CEDIP COGED CEBLEM IBGE INCRA INPI ANS DPU AGU AFT ENAC EBOK EBSP TJDFT TRF5 USP ESFCEX EAD CCR ECT PDF PDFS FEFC FP BB GSTI EC GAB ECT111 EC111').Split(' ')
$conect=('de do da dos das e a o em para com no na ao aos as os por sob sobre entre que ou').Split(' ')
function IsSigla($w){ $k=($w -replace '[^\p{L}\d]','').ToUpper(); return ($siglas -contains $k) }
function AllCapsWord($w){ $clean=$w -replace '[^\p{L}]',''; return ($clean.Length -ge 3 -and ($clean -ceq $clean.ToUpper())) }
function TitleCase($n){
  $words=$n -split '(\s+)'; $primeira=$true
  $res=foreach($w in $words){
    if($w -match '^\s+$'){ $w; continue }
    if(IsSigla $w){ $primeira=$false; $w.ToUpper() }
    elseif($w -match '\d'){ $primeira=$false; $w }
    elseif((-not $primeira) -and ($conect -contains $w.ToLower())){ $w.ToLower() }
    else { $primeira=$false; $mm=[regex]::Match($w,'\p{L}'); if($mm.Success){ $i=$mm.Index; $w.Substring(0,$i)+$w.Substring($i,1).ToUpper()+$w.Substring($i+1).ToLower() } else { $w } }
  }
  $res -join ''
}
$dirs=Get-ChildItem -LiteralPath $base -Recurse -Directory -Force -EA SilentlyContinue | Where-Object { $_.FullName -notlike "$excl*" }
$caps=$dirs | Where-Object {
  $core=$_.Name -replace '^\d+\s*-\s*',''
  (($core -split '\s+') | Where-Object { (AllCapsWord $_) -and -not (IsSigla $_) }).Count -gt 0
}
$caps=$caps | Sort-Object { ($_.FullName -split '\\').Count } -Descending
$plan=@()
foreach($d in $caps){ $nv=TitleCase $d.Name; if($nv -cne $d.Name){ $plan+=[pscustomobject]@{Full=$d.FullName;Nome=$d.Name;Novo=$nv} } }
Write-Output ("Folders a renomear: {0}" -f $plan.Count)
$plan | Select-Object -First 30 | ForEach-Object { Write-Output ("   {0}  ->  {1}" -f $_.Nome,$_.Novo) }
if($Execute){
  $ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $log="$out\log_capitaliza_$ts.csv"
  $rows=New-Object System.Collections.Generic.List[object]; $ok=0;$err=0
  foreach($p in $plan){
    if(-not (Test-Path -LiteralPath $p.Full)){ continue }
    $pai=[IO.Path]::GetDirectoryName($p.Full); $dest=Join-Path $pai $p.Novo
    try{
      if($p.Full.ToLower() -eq $dest.ToLower()){ $tmp=Join-Path $pai ($p.Novo+'~capt'); [IO.Directory]::Move($p.Full,$tmp); [IO.Directory]::Move($tmp,$dest) }
      else { if(Test-Path -LiteralPath $dest){ throw 'colide' }; [IO.Directory]::Move($p.Full,$dest) }
      $rows.Add([pscustomobject]@{De=$p.Full;Para=$dest}); $ok++
    }catch{ $err++ }
  }
  $rows|Export-Csv $log -NoTypeInformation -Encoding UTF8
  Write-Output ("`nRenomeados: {0} | Erros: {1} | Log: {2}" -f $ok,$err,$log)
}