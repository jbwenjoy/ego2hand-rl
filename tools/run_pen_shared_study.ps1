param([Parameter(Mandatory)][string]$PythonExe,[Parameter(Mandatory)][string]$TrainingOut,[Parameter(Mandatory)][string]$EvalOut)
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
Set-Location (Split-Path $PSScriptRoot -Parent)
if((Test-Path $TrainingOut) -or (Test-Path $EvalOut)){throw 'Fresh training and evaluation directories are required'}
& $PythonExe -X utf8 -u tools/train_pen_multiclip_ppo.py $TrainingOut --execute
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
if(!(Test-Path "$TrainingOut/report.json")){throw 'Missing terminal matched training result'}
& $PythonExe -X utf8 -u tools/evaluate_pen_shared_native.py $TrainingOut $EvalOut
exit $LASTEXITCODE
