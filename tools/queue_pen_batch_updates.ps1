param([int]$TrainerId,[string]$PythonExe,[string]$OutputDir)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$trainer=Get-Process -Id $TrainerId -ErrorAction SilentlyContinue
@{state='waiting_for_support_training_and_final_evaluation';pid=$TrainerId;queued=(Get-Date -Format o)}|ConvertTo-Json|Set-Content "$OutputDir/status.json"
if($trainer){$trainer.WaitForExit()}
if(!(Test-Path 'runs/pen_scaled_support_matched_v1/report.json') -and !(Test-Path 'runs/pen_scaled_support_matched_v1/failure.json')){throw 'Missing terminal support-run record'}
& $PythonExe -X utf8 -u tools/run_pen_batch_updates_matrix.py $OutputDir
exit $LASTEXITCODE
