param(
    [string]$Manifest = 'D:\workspace\supermimic-c2-training-study\configs\confirmatory_queue_v1.json',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$studyRoot = 'D:\workspace\supermimic-c2-training-study'
$studyPython = 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe'
$queueRunner = Join-Path $studyRoot 'scripts\run_matched_training_queue.py'
& $studyPython $queueRunner --manifest $Manifest --validate-only
if ($LASTEXITCODE -ne 0) { throw 'Frozen CPU queue preflight failed; no child started.' }
if (-not $Execute) { return }
$frozenPlan = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($frozenPlan.formal_training_authorized -ne $true) { throw 'Formal training is not authorized in this frozen manifest.' }
if ($frozenPlan.status -ne 'frozen_ready_for_authorization' -or $frozenPlan.caps_frozen_from_seed0_smoke -ne $true) { throw 'Draft queue cannot execute.' }
$launchDirectory = Join-Path $studyRoot 'outputs\confirmatory_queue_launch_v1'
if (Test-Path -LiteralPath $launchDirectory) { throw 'Launcher output exists; no automatic relaunch or resume.' }
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
# Each argument is quoted independently; no shell interprets manifest content.
$queueArgs = '-u "' + $queueRunner + '" --manifest "' + (Resolve-Path -LiteralPath $Manifest).Path + '" --execute'
$ownedQueue = Start-Process -FilePath $studyPython -ArgumentList $queueArgs -WorkingDirectory $studyRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
[ordered]@{ launcher_utc = [DateTime]::UtcNow.ToString('o'); queue_launcher_pid = $ownedQueue.Id; manifest_path = (Resolve-Path -LiteralPath $Manifest).Path; manifest_sha256 = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLower(); scope = 'Explicitly authorized detached queue; Goal pause does not terminate it. Actual child PID/OS CreationDate is recorded by queue state.' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDirectory 'launch.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $launchDirectory 'launch.json') -Raw
