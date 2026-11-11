# Research plan v0.52: fresh interventions and matched response supervision

2026-11-11. Previous goal turn was progress: native CHECK probes and source were
committed as a78d124, the original WM reached1000 with12/12 structural checks,
and the requested checkpoint pause was observed. The user resumed the Goal.
This stage addresses response identifiability before using WM action gradients
for a reference-conditioned Actor. No old check source is promoted into fitting.

## Recorded1000 result

990 resumed updates took2.14309s. Old CHECK normalized value MSE was.360313 for
delta52,.240617 for wrench6 and.613323 for Fn3. TRAIN delta52 MSE.005119 shows
that a good training fit alone is insufficient. On the old paired cases,
delta52 response NMSE versus the zero-response baseline was1.054118, contact9
.721434. At native action amplitude.01, AD delta52 response NMSE was1.138616.
Thus the original1000 model is saved and healthy, but not qualified for reliable
imagined physical control. The old nvidia-smi file-loop attempt had zero samples;
there is no inferred utilization benefit from those missing measurements.

## New causal data, already acquired before fitting

`configs/mug_interventions_v1.toml` fixes new TRAIN seeds200..211 and new CHECK
seeds8200..8203; none is an old acquisition seed. New prefixes use the same
bounded joint-feedback/multisine rule at original nominal physics/reset. At
k2/7/12, restore the full numerical/controller source and perturb every joint
by±.04 and±.2 about its current profile action. Actual executed control is the
input and derivative denominator, including asymmetric room near a limit.

|Role|Prefix states|Rows|Pairs|Native dt|
|---|---:|---:|---:|---|
|New TRAIN|36|2880|1440|.0003125|
|New CHECK|24|1920|960|.0003125 and.00015625|

397824 native steps took174.589s, including prefixes and replay barriers. All60
full-state/controller replay barriers passed. All branches are complete20ms;
no partial/terminal branch was silently discarded or replaced. Raw full states,
controls, endpoint charts, per-tick contacts/frames/forces and trace ranges are
retained. Prefix and center-replay rows are evidence only, excluded from fitting.

The independent reviewer reconstructed2123225 raw contact records. Its8/8 checks
passed: controller error0, endpoint chart/observer error0, force/couple/moving-COM
wrench/Fn error at most8.88e-15; TRAIN/CHECK source overlap0 and old diagnostic
source overlap0. Pair validation includes complete native-state equality and
single positive actual-u difference, keyed by seed/dt/k/joint/amplitude.

This is one reset, one object and a narrow prefix distribution. Independent
prefix seeds are not unseen trajectories, independent RGB reconstruction,
source-human contact truth, calibrated materials or successful demonstrations.

## Matched experiment fixed before fitting

`configs/mug_paired_response_first100.toml` defines two arms, ABS and CF. Both
warm-start identical model AND Adam1000 from the saved parent. Original TRAIN
scales remain frozen, with no refit. Both uniformly sample the same3258 absolute
TRAIN rows (old378 + new2880), and the same1440 TRAIN pairs, using matching new
CUDA sampler seed20261111. Batch256 absolute rows and64 pairs, same forwards,
optimizer type, learning rate, clipping and update budget. No CHECK selection.

ABS has coefficient0 and CF coefficient1 on the extra pair loss. Even ABS runs
the pair forwards, so the comparison has matched row/forward/update counts.
For each normalized output, compare the two predicted differences with the
native difference, divide by actual positive `u_plus-u_minus`, and multiply by
.025rad. No epsilon denominator or assumed symmetric action mapping. The loss
weights hand40 and object12 by.5 each, warm26 by.1, wrench6 by.2 and Fn3 by.1.
The original absolute loss and architecture are unchanged. This isolates this
response-supervision change, not CHORD reward or an Actor A1/A2 result.

Planned stage budget is2000 paired updates. The FIRST paired checkpoint is100,
with each arm's Adam step1100. At that point save all models/optimizers/scales,
RNG/samplers, gradients, source/data/config pins and logs atomically. Check full
raw readback, next batches/predictions and matching sampler cursors. Evaluate
new CHECK actual-u AD and finite secants, plus unchanged old diagnostics.
Estimate remaining1900 paired-update compute from the last50 CUDA events,
then PAUSE Goal at the first healthy checkpoint, as requested. Do not continue
optimization merely to improve a check result.

GPU data and targets are resident. No geometry/native/host health checks occur
inside updates. New telemetry uses a separate thread with one-shot nvidia-smi
queries, avoiding unflushed short file-loop samples; report query windows,
failures and device-wide scope. Tiny-workload utilization is not global training
speedup. Include data collection/review and future compilation/retry costs in
any eventual shared-versus-per-trajectory cost claim.

## After the next explicit resume

Continue the fixed paired budget if structural health permits, keeping CHECK
read-only. Response NMSE below zero-prior1 is a diagnostic, not a sufficient
qualification threshold. Examine object and contact direction/cosine, magnitude,
unseen prefixes, finite-width/dt uncertainty and multi-step/no-aid consequences.
Only then rely on these derivatives for reference-conditioned feedback/Actor
training, native rollouts and matched task-level contact-goal comparisons.

Retain the FULL goal: fixed-wrist cube/pen loading/playback, short learning and
save/resume/no-aid evaluation; object-frame wrench support and matched A1/A2;
collision/inertia/material/gravity/input/controller contracts; independent RGB
ego/full trajectory reconstruction; multiple trajectories and total shared-vs-
per-trajectory cost; hardware subversion/joint/zero/gain/latency/observations.
No driver/CUDA changes, no duplicate top-level study directory, no full success
claim from data acquisition or a structurally healthy response checkpoint.

Evidence: `runs/mug_response_resume1000_v1/health_1000.json`,
`runs/mug_interventions_v1/report.json`,
`runs/mug_interventions_packed_v1/report.json`. Actual new training status belongs
to `runs/mug_paired_response_v1/health_0100.json`, if that independent report exists.
