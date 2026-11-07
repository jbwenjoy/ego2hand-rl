param(
    [string]$Manifest = 'D:\workspace\supermimic-chord-study\configs\replication-seeds123-queue-v1.json',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$plan = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$runnerPath = Join-Path $PSScriptRoot 'run_matched_training_queue.py'
& $plan.python_executable $runnerPath --manifest $manifestPath --validate-only
if ($LASTEXITCODE -ne 0) { throw 'Queue CPU preflight failed' }
if (!$Execute) { return }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant() -ne $plan.launcher_sha256) { throw 'Launcher hash changed' }
$outputPath = $plan.queue_output
if (Test-Path -LiteralPath (Join-Path $outputPath 'state.json')) { throw 'Queue already started; no automatic restart' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'queue_stdout.log'
$stderrPath = Join-Path $outputPath 'queue_stderr.log'
$record = [ordered]@{
    status = 'launching'
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    manifest = $manifestPath
    manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
    queue_runner_sha256 = $plan.queue_runner_sha256
    stdout = $stdoutPath
    stderr = $stderrPath
    lifecycle = 'Detached, hidden, single-GPU serial queue; Goal pause leaves queue and active training running'
}
$queueProcess = Start-Process -FilePath $plan.python_executable -ArgumentList @('-u', ('"' + $runnerPath + '"'), '--manifest', ('"' + $manifestPath + '"'), '--execute') -WorkingDirectory (Split-Path $PSScriptRoot -Parent) -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record.status = 'detached_started'
$record.launcher_pid = $queueProcess.Id
$record.launcher_start_utc = $queueProcess.StartTime.ToUniversalTime().ToString('o')
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding utf8
$record | ConvertTo-Json -Depth 6
