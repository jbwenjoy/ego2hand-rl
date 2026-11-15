# Training2/contact collector and initial EMA probe — 2026-11-16

The owned runtime now has a separately tagged complete training2 collector/state.
Final integration83/83, independent continuation84/84 and322 tests pass. The
loaded learner actually collects another4 controls, updates, saves and reloads;
this is more than a network-only readback. All updated weights in this round
are integration DIAGNOSTICS, not policy candidates. Latest trained-policy video
and failed original strict task results remain unchanged.

## Shared collector and complete state

learning/owned_pen_ppo.py subclasses the maintained NativePenPPO. It reuses the
existing collector, network, PPO update and GAE. During collection an exclusive
temporary step callback copies every borrowed contact-window GPU array into
owned rollout storage BEFORE the base collector resets terminal worlds. The
callback is removed even on errors. Native/cache guards run at block boundaries.
Each row stores maxima, Fn6/nonhand sums,8-frame wrench sums, sample/contact
counts, source axes/COM, clocks, error flags and the USED reference phase.
These extra columns do not change the161/188 observations or tracking reward.

Schema `pen_owned_contact_PPO_training:2` contains native4, modules, AdamW,
frozen normalizers, previous raw normalization batch, observations, complete
rollout/contact buffers and all training/capture/phase counters. The native4
schema, previous training1 loaders and default controller files are unchanged.
Old/new training schemas reject implicit conversion. No old weights-only
initialization is selected in this diagnostic; networks start fresh.

Preflight validates layout/types/counters, intrinsic applied-action/phase/
timeout/contact contracts and raw normalization-batch identity. It reconstructs
saved observations from canonical native/Python source paths using the SAME
feature encoders and frame-ID property getters on CPU, without changing live
state. It stages module/optimizer resources under a forked RNG scope before
native restoration. Corrupted observations, contact counts, normalization
batch and Adam step shape reject without changing native/learner/RNG digests.
Saved optimizer step tensors are copied independently; no caller-state alias.

CPU/GPU feature arithmetic is not assumed bitwise: initial maximum differences
are2.384e-7(policy)/3.815e-6(critic), and resumed3.815e-6..7.629e-6 critic
roundoff fits declared2e-5 absolute plus1e-6 relative tolerance. Actual loaded
GPU observations, buffers, modules, optimizer and native state read back exactly.
The normalizer count is288 after calibration/first rollout and416 after the
loaded next rollout. A completed rollout can be updated once; frozen-policy
logprob/value consistency is checked BEFORE updating, not against stale records
after parameters change. This is not physical W32 future-repeat qualification.

## Observed integration

Each qualification has32 worlds x8 calibration controls and4 sampled controls,
then one32-minibatch diagnostic update. A separate W1 pre-start final-frame
cohort runs1 control: real timeout bootstrap and all8 contact samples survive
the subsequent reset. Calibration data/cost are included. The final run records
32 raw native packets(1024 world-integrations) for independent NumPy folding.
All window fields match; largest wrench-sum difference8.882e-16, other maximum
differences0. The two collector consistency cohorts each pass22/22 checks.

The first v1 qualifier reported83/83 but its file-immutability expression compared
a file read to itself. v2 saves the pin BEFORE negative loads and compares it
afterwards, then redoes the entire qualification. v1/source/outputs remain;
its uninformative check is not treated as independent file-integrity evidence.

Fresh-process review v1 passes51/51, including two identical fixed-rollout CUDA
update clones and source/live-state immutability. Extended independent review
v2 passes84/84: the same fixed-data checks, a real loaded4-control continuation,
one update, and complete reload. It observes iteration2/64 optimizer steps/
256 training transitions and416 normalization samples. New physical trajectories
are not compared bitwise; W32 future numerical repeatability is still open.

Current reward remains geometry-unpenalized and is used ONLY for state/update
integration. The4x32 primary pre-update controls show:

| Original geometry subgate | Maximum | Violating control rows /128 |
|---|---|---|
|Object penetration <=2mm|11.9006mm|55(26 worlds)|
|Robot self penetration <=.2mm|5.71253mm|28(13 worlds)|
|Joint excursion <=.001rad|0|0|
|Object nonhand normal <=.1N|0|0|

This is early single-reference data, not an expert or qualified policy rollout.
Do not continue epoch sweeps on this unchanged reward and call survival success.

## Initial residual/EMA feasibility probe

A separate never-fit W1 probe restores the SAME native4 source for18 short
episodes: ZERO_RAW, HOLD_INITIAL_RAW, and hold plus Gaussian noise std
.05/.1/.2/.5, with three fixed noise seeds. ZERO/HOLD repeats have identical
inputs; they are repeated source diagnostics, not independent training seeds.
All18 run4 original-reference controls(.08s); no retiming/pose reset inside an
episode, optimizer or object aid. The8 probe accounting checks pass.

Initial EMA max magnitude is.0958996rad; initialized raw max.191799<1. Under the
original scale.5/EMA.5, new_EMA=.5*old_EMA+.25*clipped_raw. Thus zero raw halves
the initialized offset, while raw=2*initial_EMA keeps it constant. Both recurrences
are measured, not inferred as a policy improvement.

| Deterministic probe, common.08s | Object/self max | Max control-end COM / full angle |
|---|---|---|
|Zero raw|1.97583mm /0|18.228mm /47.1504deg|
|Hold initialized raw|1.97668mm /0|4.25631mm /12.0595deg|

Hold improves this source's short control-end tracking, but its full angle still
exceeds the original10deg maximum. These are4 control-end errors, not a dense
native-step RMS/full7.36s strict evaluation. Zero and hold both pass geometry on
this prefix, so EMA decay is NOT established as the cause of penetration.
All hold+noise std<=.2 prefixes pass geometry on these three noise seeds; std.5
fails self penetration in3/3(1.829..2.752mm), with one object3.49956mm. This is
not a universal safe-exploration threshold. No zero+noise factorial comparison
or matched W32 policy-noise causal result is claimed. All full-task flags remain
false. These records are not promoted into experts or world-model training.

## Costs, evidence and next step

| Work | New world-integrations | Diagnostic optimizer steps |
|---|---|---|
|Qualification v1|3080|32|
|Final qualification v2|3080|32|
|Independent review v1|0|64 clones|
|Extended review v2|1024|64 clones +32 continued|
|EMA probe|576|0|
|Total|7760|224;0 retained candidate updates|

Sources are frozen before every GPU job; original snapshots, all buffers,
checkpoints and initial check version are preserved. Compact evidence:
provenance/owned_ppo_results/ledger.json. The integrated path was NOT timed as a
performance study; raw CPU logging and cold checks prevent a training-speedup
claim. No new actor video/geometry reward/support advantage/WM-gradient/ego/full
task success is declared. Next: preregister paired initialization/exploration
and geometry rules plus correctly timed BODY support targets before learning.
