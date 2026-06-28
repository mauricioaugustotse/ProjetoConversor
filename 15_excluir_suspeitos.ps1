# Exclui (para a Lixeira) os arquivos suspeitos detectados (temporarios, vazios, corrompidos). Registro ja em suspeitos_exclusao.csv.
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName Microsoft.VisualBasic
$out = 'C:\Users\mauri\ProjetoConversor'
$susp = Import-Csv "$out\suspeitos_exclusao.csv" -Encoding UTF8
$ok = 0; $err = 0; $errs = New-Object System.Collections.Generic.List[string]
foreach($s in $susp){
  if(Test-Path -LiteralPath $s.FullName){
    try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($s.FullName, 'OnlyErrorDialogs', 'SendToRecycleBin'); $ok++ }
    catch { $err++; if($errs.Count -lt 20){ $errs.Add($s.FullName) } }
  }
}
Write-Output ("Excluidos para a Lixeira: {0:N0} | Erros: {1:N0}" -f $ok, $err)
$errs | ForEach-Object { Write-Output "  ERRO: $_" }