# Research plan v0.51: response fitting and native directional checks

2026-11-08. The previous goal turn made progress: e128161 committed causal native
H20 data and the new response model; checkpoint10 was saved,12/12 structural
checks passed and the Goal was paused at the requested boundary. The user has
now resumed it. This stage continues the same predeclared1000-update recipe,
without restarting Adam, refitting scales or selecting a model on CHECK data.

## Evidence before the resumed fitting

- New CHECK-only native probes:144 cases/72 central pairs, amplitudes.01/.05/.2,
  joints0/4/8/9 at baseline k0/3/6, both original dt.14400 native steps including
  six replay barriers. All six zero-action replays exactly match saved native
  state, observations, warmstarts and mean contact labels. Raw per-tick traces,
  contact records, complete numerical states and actual controls are retained.
- Probe inputs share the full native/controller source within each central
  pair. Actual `du/da=.025 rad` for these four symmetric-room joints; asymmetric
  limit kinks are explicitly rejected. These probes never fit or select weights.
- The checkpoint10 model's small-amplitude delta52 AD-versus-native response
  NMSE is1.00229, cosine.00338; contact9 NMSE.97519. Thus healthy serialization
  does not imply a useful response derivative. Native coarse/fine small-amplitude
  delta52 response cosine.999745, contact9.999418. These compare baseline prefixes
  at different dt, not identical cross-dt numerical states or a rigorous native
  discretization-error bound. Finite contact secants are not analytic derivatives.
- New continuation tests cover exact next optimizer/batch/gradient replay,
  noncommuting world-left rotation labels, rejection of mismatched pair sources,
  known-linear response JVP/EMA chain rule and denominator handling.123 tests pass.

Evidence is in `runs/mug_action_probes_v1/report.json`,
`runs/mug_response_check10_v1/report.json`, and
`runs/mug_response_resume_tests123.log`. Current training status belongs to its
independent health artifact; this document does not assert it has already run.

## Authorized bounded continuation

`configs/mug_response_resume1000.toml` resumes checkpoint10's exact model, Adam,
TRAIN scales, RNG and CUDA sampler, retaining original architecture, loss,
batch256, learning rate and dataset roles. There are990 new updates11..1000.
The first checkpoint of this resumed stage is1000; no intermediate checkpoint
is created and no later optimization is performed. Native probes and old48
paired-response cases are read-only diagnostics. No Actor is trained.

Resident TRAIN tensors feed the common response update function. No geometry,
native integration, health scalar transfer or validation runs inside the update
loop. Device-wide nvidia-smi utilization/VRAM is sampled asynchronously; these
samples include other desktop workloads and are not process-only occupancy.
This small single-task MLP workload cannot establish an overall SuperMimic
speedup or justify increasing batch size merely to make GPU utilization larger.

Save full state atomically, verify model/optimizer bytes, Adam1000, RNG, sampler,
CHECK predictions, frozen scales and data lineage. Report TRAIN/CHECK absolute
prediction errors, old paired responses and new AD/finite-secant comparisons,
including object12 and contact9 blocks. Timing reports actual990-update wall
time and last100 CUDA-event median. At1000 the fixed fitting budget has no
updates remaining; this is not an ETA for native data collection or research.
After the first healthy resumed-stage checkpoint/report, pause Goal as requested.

## Work retained after the next explicit resume

If the response remains unreliable, propose fresh TRAIN interventions with
separate acquisition seeds/states and paired-action coverage; do not promote
these CHECK probes into supervision. Preserve bad response results rather than
tuning repeatedly against them. Useful one-step values are insufficient for
autoregressive stability, solver-memory prediction or imaginary reward gradients.

Then establish reference-conditioned physical feedback/Actor observations,
validate short-horizon no-aid dynamics and contact/action directions, perform
matched A1/A2 and full trajectory evaluation. The complete controlled cube/pen,
object-frame support, ego collision/inertia/material/gravity/input contracts,
independent RGB reconstruction, multi-trajectory shared-versus-per-trajectory
costs and hardware version/zero/gain/latency/observation requirements remain.
Neither the current data nor a response checkpoint proves holding success,
CHORD benefit, real-video generalization or completion of the full Goal.

No driver/CUDA changes or new workspace-level study directories.
