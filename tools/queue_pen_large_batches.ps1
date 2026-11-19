param([Parameter(Mandatory)][int]$TrainerId,[Parameter(Mandatory)][string]$PythonExe,[Parameter(Mandatory)][string]$OutputDir)
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
Set-Location (Split-Path $PSScriptRoot -Parent)
$trainer=Get-Process -Id $TrainerId -ErrorAction SilentlyContinue
if(Test-Path $OutputDir){throw 'Output directory already exists'}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
@{state='waiting_for_training_and_evaluation';training_pid=$TrainerId;worlds=@(256,512,1024,2048,4096);queued=(Get-Date -Format o)} | ConvertTo-Json | Set-Content "$OutputDir/status.json"
if($trainer){$trainer.WaitForExit()}
$dir='runs/pen_scaled_train_v1'
if(!(Test-Path "$dir/report.json") -and !(Test-Path "$dir/failure.json")){throw 'Trainer exited without a terminal report. Inspect the run before starting GPU benchmarks.'}
$summary=@()
foreach($w in 256,512,1024,2048,4096){
    @{state='benchmarking';worlds=$w;started=(Get-Date -Format o)} | ConvertTo-Json | Set-Content "$OutputDir/status.json"
    & $PythonExe -X utf8 -u tools/benchmark_pen_large_batches.py $w "$OutputDir/W$w" 1> "$OutputDir/W$w.stdout.log" 2> "$OutputDir/W$w.stderr.log"
    $code=$LASTEXITCODE
    if($code -eq 0 -and (Test-Path "$OutputDir/W$w/report.json")){
        $r=Get-Content "$OutputDir/W$w/report.json" -Raw | ConvertFrom-Json
        if($w -eq 256){$baseline=$r.valid_steps_s}
        $ratio=if($baseline){$r.valid_steps_s/$baseline}else{$null}
        $summary+=@{worlds=$w;state='passed';effective_steps_s=$r.valid_steps_s;vs_256=$ratio;raw_steps_s=$r.raw_steps_s;valid_fraction=$r.valid_fraction;mean_W=$r.board_mean_W;peak_W=$r.board_peak_W;peak_MiB=$r.board_peak_MiB;whole_run_peak_MiB=$r.whole_run_board_peak_MiB}
    }else{
        $summary+=@{worlds=$w;state='failed';exit_code=$code;stderr="W$w.stderr.log"}
    }
    ConvertTo-Json -InputObject $summary | Set-Content "$OutputDir/results.json"
}
@{state='finished';ended=(Get-Date -Format o)} | ConvertTo-Json | Set-Content "$OutputDir/status.json"
