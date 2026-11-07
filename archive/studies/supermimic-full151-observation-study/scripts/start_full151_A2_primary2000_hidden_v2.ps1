$ErrorActionPreference = 'Stop'
$studyRoot = 'D:\workspace\supermimic-full151-observation-study'
$selectedPython = 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe'
$supervisorPath = Join-Path $studyRoot 'scripts\run_full151_learning_v3.py'
$configPath = Join-Path $studyRoot 'configs\full151-v3-primary2000-A2-seed0-v1.json'
$launchDir = Join-Path $studyRoot 'outputs\full151_v3_primary2000_A2_launch_v1'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.method -cne 'A2' -or $config.role -cne 'primary_matched' -or !$config.GPU_run_authorized -or $config.automatic_retry) { throw 'Frozen A2 primary authorization required' }
if (Test-Path -LiteralPath $launchDir) { throw 'Launch directory exists: retain its PID; never launch twice' }
if (Test-Path -LiteralPath $config.output) { throw 'Worker output already exists; never restart' }
New-Item -ItemType Directory -Path $launchDir | Out-Null
$launchedUtc = [DateTime]::UtcNow.ToString('o')
$ownedLauncher = Start-Process -FilePath $selectedPython -ArgumentList @('-u', ('"' + $supervisorPath + '"'), '--config', ('"' + $configPath + '"'), '--run') -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $launchDir 'supervisor.stdout.log') -RedirectStandardError (Join-Path $launchDir 'supervisor.stderr.log') -PassThru
$launchRecord = [ordered]@{
    schema = 'full151_owned_hidden_supervisor_launch:2'
    launched_utc = $launchedUtc
    retained_launcher_pid = [int]$ownedLauncher.Id
    method = 'A2'
    supervisor = $supervisorPath
    config = $configPath
    config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    worker_sha256 = (Get-FileHash -LiteralPath (Join-Path $studyRoot 'scripts\execute_full151_learning_v3.py') -Algorithm SHA256).Hash.ToLowerInvariant()
    supervisor_sha256 = (Get-FileHash -LiteralPath $supervisorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    hidden = $true
    automatic_retry = $false
    intended_scope = 'One A2 from-scratch primary only. Continue owned training after authorized Goal pause; A1L/P_ONLY are preregistered but not queued.'
}
$launchRecord | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $launchDir 'launch_started.json') -Encoding utf8
$watchPath = $config.output + '_watchdog\state.json'
$initialRows = $null
for ($attempt = 0; $attempt -lt 35; $attempt++) {
    if (Test-Path -LiteralPath $watchPath) {
        $watch = Get-Content -LiteralPath $watchPath -Raw | ConvertFrom-Json
        if ($watch.supervisor_pid -and $watch.launcher_pid -and $watch.actual_pid -and $watch.actual_process_OS_observation.status -ceq 'exact_OS_identity_observed') {
            $ownedIds = @([int]$ownedLauncher.Id, [int]$watch.supervisor_pid, [int]$watch.launcher_pid, [int]$watch.actual_pid)
            $initialRows = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $ownedIds } | ForEach-Object { [pscustomobject]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;creation_utc=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine;executable_path=$_.ExecutablePath} })
            if ($initialRows.Count -ne 4) { throw 'Owned process chain missing: preserve launch_started PID and investigate without relaunch' }
            break
        }
        if ($watch.status -and $watch.status -cne 'running') { throw ('Supervisor terminal before initial identity: ' + $watch.status) }
    }
    Start-Sleep -Milliseconds 500
}
if ($null -eq $initialRows) { throw 'Initial identity observation timed out; retained launch PID remains authoritative; never relaunch' }
$initialPath = Join-Path $launchDir 'initial_CIM.json'
ConvertTo-Json -InputObject $initialRows -Depth 8 | Set-Content -LiteralPath $initialPath -Encoding utf8
$launchRecord.initial_CIM = @{path=$initialPath;sha256=(Get-FileHash -LiteralPath $initialPath -Algorithm SHA256).Hash.ToLowerInvariant()}
$launchRecord | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $launchDir 'launch.json') -Encoding utf8
$launchRecord | ConvertTo-Json -Depth 12
