param([Parameter(Mandatory=$true)][string]$RunName, [double]$Minutes = 30)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$runDir = Join-Path $PSScriptRoot "logs/$RunName"
New-Item -ItemType Directory -Force $runDir | Out-Null
$statePath = Join-Path $runDir 'experiment-state.json'
$started = Get-Date
function Set-State([string]$Phase, [string]$Details = '') {
    @{phase=$Phase; details=$Details; pid=$PID; started=$started.ToString('o'); updated=(Get-Date).ToString('o'); minutes=$Minutes} |
        ConvertTo-Json | Set-Content $statePath -Encoding utf8
}
function Run-Uv([string]$LogName, [string[]]$CommandArgs) {
    & "$PSScriptRoot/uv-local.ps1" @CommandArgs *> (Join-Path $runDir $LogName)
    if ($LASTEXITCODE -ne 0) { throw "$LogName failed with exit code $LASTEXITCODE" }
}
try {
    $baseline = 'logs/windows_warm_cache_resume/ckpt_latest.pt'
    Set-State 'baseline_evaluation'
    Run-Uv 'baseline-eval.log' @('run','--locked','python','scripts/eval_benchmark.py','--ckpt',$baseline,'--num-envs','32','--out',"$runDir/baseline-benchmark.json")
    Set-State 'training'
    Run-Uv 'train.log' @('run','--locked','sm-train','--run-name',$RunName,'--resume',$baseline,
        '--seed','1','--env.num-envs','128','--env.replay-capacity','128',
        '--wm-trainer.batch-size','128','--wm-trainer.batches-per-update','10',
        '--policy-trainer.batch-size','128','--policy-trainer.critic-epochs','5',
        '--policy-trainer.critic-minibatches','2',
        '--schedule.total-iterations','20000','--schedule.max-hours',([string]($Minutes/60)),
        '--schedule.log-interval','100','--schedule.checkpoint-interval','1000')
    $checkpoint = "$runDir/ckpt_latest.pt"
    Set-State 'final_evaluation'
    Run-Uv 'final-eval.log' @('run','--locked','python','scripts/eval_benchmark.py','--ckpt',$checkpoint,'--num-envs','32','--out',"$runDir/final-benchmark.json")
    Set-State 'rendering'
    Run-Uv 'record.log' @('run','--locked','python','scripts/media/record_rollout.py','--ckpt',$checkpoint,'--npz-dir','data/npz/lafan1',
        '--clips','walk1_subject1','dance1_subject1','jumps1_subject1','--out',"$runDir/rollouts.npz")
    Run-Uv 'render.log' @('run','--locked','python','scripts/media/render_rollout.py','--traj',"$runDir/rollouts.npz",'--out-dir',"$runDir/media",'--seconds','10','--width','960','--height','540','--concat',"$runDir/preview.mp4")
    Set-State 'complete'
} catch {
    Set-State 'failed' $_.Exception.Message
    $_ | Out-String | Set-Content (Join-Path $runDir 'error.txt')
    exit 1
}
