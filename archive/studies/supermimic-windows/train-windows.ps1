param(
    [switch]$Eager,
    [int]$Steps = 160,
    [int]$NumEnvs = 128,
    [int]$BatchSize = 128,
    [string]$RunName = '',
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
if (-not $RunName) {
    $modeName = if ($Eager) { 'eager' } else { 'compiled' }
    $RunName = 'windows_' + $modeName + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
}
$runArgs = @('run', '--locked', 'python', 'scripts/windows_validate.py',
    '--steps', $Steps, '--num-envs', $NumEnvs, '--batch-size', $BatchSize,
    '--run-name', $RunName)
if (-not $Eager) { $runArgs += '--compiled' }
if ($Resume) { $runArgs += @('--resume', $Resume) }
& "$PSScriptRoot\uv-local.ps1" @runArgs
exit $LASTEXITCODE
