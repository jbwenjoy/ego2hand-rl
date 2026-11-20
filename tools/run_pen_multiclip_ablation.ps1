param([Parameter(Mandatory)][string]$PythonExe,[Parameter(Mandatory)][string]$Partial,[Parameter(Mandatory)][string]$Mixed,[Parameter(Mandatory)][string]$Eval)
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach($p in @($Partial,$Mixed,$Eval)){if(Test-Path $p){throw "Output occupied $p"}}
& $PythonExe -X utf8 -u tools/train_pen_multiclip_partial.py $Partial --execute
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $PythonExe -X utf8 -u tools/train_pen_balanced.py $Mixed
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $PythonExe -X utf8 -u tools/evaluate_pen_mixed_updates.py $Partial $Mixed $Eval
exit $LASTEXITCODE
