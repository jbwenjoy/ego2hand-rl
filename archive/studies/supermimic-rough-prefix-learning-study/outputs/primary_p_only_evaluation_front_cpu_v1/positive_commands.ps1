$ErrorActionPreference = 'Stop'
$study = 'D:\workspace\supermimic-rough-prefix-learning-study'
$proofRoot = Join-Path $study 'outputs\primary_p_only_evaluation_front_cpu_v1'
$python = 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
$sourceFiles = [ordered]@{
  builder = Join-Path $study 'scripts\build_primary_evaluation_config_cpu_v2.py'
  entry = Join-Path $study 'scripts\evaluate_rough_prefix_pen_v2.py'
  supervisor = Join-Path $study 'scripts\run_rough_evaluation_v2.py'
  evaluation = Join-Path $study 'src\rough_prefix\evaluation_v2.py'
  policy_probe = Join-Path $study 'src\rough_prefix\policy_probe.py'
  actor_loader = Join-Path $study 'src\rough_prefix\checkpoint_v2.py'
  protocol = Join-Path $study 'protocols\rough-evaluation-v3.json'
}
$sourcePins = [ordered]@{}
$commands = [Collections.Generic.List[object]]::new()
$checks = [ordered]@{}
$configPath = Join-Path $study 'configs\p_only-primary-policy-evaluation-v1.json'
$healthPath = Join-Path $study 'outputs\health_primary_p_only_final_v1\report.json'
$actorPath = Join-Path $study 'outputs\actual_actor_load_primary_p_only_final_v1\report.json'
$trainingPath = Join-Path $study 'configs\p_only-primary_matched-authorized_v1.json'
function FilePin([string]$Path) {
  return [ordered]@{ path = (Resolve-Path -LiteralPath $Path).Path; sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower() }
}
function InvokeCpu([string]$Name, [string]$Script, [string[]]$ExtraArguments) {
  if ($ExtraArguments -contains '--run') { throw 'No watch/GPU allowed in CPU preparation' }
  $start = [Diagnostics.Stopwatch]::StartNew()
  & $python $Script @ExtraArguments 1> (Join-Path $proofRoot ($Name + '.stdout.json')) 2> (Join-Path $proofRoot ($Name + '.stderr.txt'))
  $code = $LASTEXITCODE
  $start.Stop()
  $commands.Add([ordered]@{name=$Name;command=@($python,$Script)+$ExtraArguments;exit_code=$code;elapsed_seconds=$start.Elapsed.TotalSeconds})
  $checks[$Name + '_exit0'] = $code -eq 0
  if ($code -ne 0) { throw ('CPU child failed: ' + $Name) }
  return Get-Content -LiteralPath (Join-Path $proofRoot ($Name + '.stdout.json')) -Raw | ConvertFrom-Json
}
$failure = $null
try {
  $snapshot = Join-Path $proofRoot 'source'
  New-Item -ItemType Directory -Path $snapshot | Out-Null
  foreach ($item in $sourceFiles.GetEnumerator()) {
    $sourcePins[$item.Key] = FilePin $item.Value
    Copy-Item -LiteralPath $item.Value -Destination (Join-Path $snapshot (Split-Path -Path $item.Value -Leaf))
  }
  if ($sourcePins.builder.sha256 -ne '831acb03c2fc7de41fb7a256a18d3cdc0f66f14c2bb82f1e2da3c32757b4def8' -or $sourcePins.protocol.sha256 -ne '5caf3fa4323c85d7ae41a4ed741a7aedf648f853613fa57a1b0f504d2cf3a6a1') { throw 'Frozen builder/protocol drift' }
  $health = Get-Content -LiteralPath $healthPath -Raw | ConvertFrom-Json
  if ($health.status -ne 'PASS_SAVED_ENGINEERING_HEALTH' -or $health.iteration -ne 2000 -or $health.final_iteration_scope -ne $true -or @($health.checks.PSObject.Properties | Where-Object { $_.Value -ne $true }).Count -ne 0) { throw 'Await final health green; do not build' }
  $checks.finalhealth_green = $true
  $built = InvokeCpu 'builder' $sourceFiles.builder @('--training-config',$trainingPath,'--health',$healthPath,'--actor-load',$actorPath,'--version','v1')
  $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
  $checks.exact_selected_Pfinal2000_Gfalse = $config.GPU_run_authorized -eq $false -and $config.method -eq 'P_ONLY' -and $config.seed -eq 0 -and $config.checkpoint_iteration -eq 2000 -and $config.checkpoint.sha256 -eq 'c73044119386d8041eaed3e28ab456b5b0a973e220773f1fddc406cb72b22bbe'
  $checks.output_reserved_not_created = -not (Test-Path -LiteralPath $config.output) -and -not (Test-Path -LiteralPath ($config.output + '_watchdog'))
  foreach ($entry in @(@('entry',$sourceFiles.entry),@('supervisor',$sourceFiles.supervisor))) {
    $given = InvokeCpu $entry[0] $entry[1] @('--config',$configPath)
    $checks[$entry[0] + '_default_CPU_only'] = $given.status -eq 'CPU_INPUT_SOURCE_PREFLIGHT_ONLY' -and $given.method -eq 'P_ONLY' -and $given.Torch_imported -eq $false -and $given.Warp_imported -eq $false -and $given.GPU_started -eq $false -and $given.physics_steps -eq 0
  }
  $protocol = Get-Content -LiteralPath $sourceFiles.protocol -Raw | ConvertFrom-Json
  $closed = $true
  foreach ($shared in $protocol.shared_executed_consumer.PSObject.Properties) {
    $actual = $config.source_pins.PSObject.Properties['protocol_shared_consumer__' + $shared.Name].Value
    if ($null -eq $actual -or $actual.path -ne $shared.Value.path -or $actual.sha256 -ne $shared.Value.sha256) { $closed = $false }
  }
  $checks.shared_six_consumers_exact_unchanged = $closed -and $protocol.shared_executed_consumer.PSObject.Properties.Count -eq 6
  $training = Get-Content -LiteralPath $trainingPath -Raw | ConvertFrom-Json
  $checks.actual_training_protocol4_preserved = $config.source_pins.training_protocol.path -eq $training.protocol.path -and $config.source_pins.training_protocol.sha256 -eq $training.protocol.sha256
  $checks.same_tested_horizon_caps_no_retry = $config.worlds -eq 1 -and $config.hard_process_seconds -eq $protocol.hard_process_seconds -and $config.soft_worker_seconds -eq $protocol.soft_worker_seconds -and $config.automatic_retry -eq $false -and $config.compile_fallback -eq $false
  $current = $true
  foreach ($item in $config.source_pins.PSObject.Properties) { if ((FilePin $item.Value.path).sha256 -ne $item.Value.sha256) { $current = $false } }
  $checks.all_current_dependency_pins = $current
  $checks.no_watch_or_GPU_authorization = $true
} catch { $failure = $_.Exception.ToString(); $checks.campaign_complete = $false }
$passed = @($checks.GetEnumerator() | Where-Object { $_.Value -eq $true }).Count
$report = [ordered]@{
  schema='rough_primary_P_ONLY_evaluation_positive_CPU_preparation:1'
  status= $(if ($passed -eq $checks.Count -and $null -eq $failure) { 'PASS_CPU_ONLY' } else { 'FAIL_OR_INCOMPLETE' })
  checks=$checks;passed=$passed;total=$checks.Count;commands=$commands
  source_pins=$sourcePins;config= $(if (Test-Path -LiteralPath $configPath) { FilePin $configPath } else { $null })
  health=FilePin $healthPath;actual_actor_load=FilePin $actorPath;error=$failure
  watch_called=$false;GPU_run_authorized=$false;Env_or_Runner_constructed=$false;physics_steps=0
  scope='Saved-only P_ONLY primary final2000 metadata, unchanged shared consumer and default CPU paths. No physics or evaluation authorization.'
}
[IO.File]::WriteAllText((Join-Path $proofRoot 'report.json'),($report | ConvertTo-Json -Depth 14),[Text.UTF8Encoding]::new($false))
$report | Select-Object status,passed,total,error | ConvertTo-Json
if ($report.status -ne 'PASS_CPU_ONLY') { exit 1 }
