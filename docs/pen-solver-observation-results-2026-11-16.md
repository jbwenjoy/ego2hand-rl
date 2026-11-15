# Fresh-process solver observation and common WM data — 2026-11-16

Two fresh processes reproduce strict-path impulse failures on selected original
TRAIN conditions. The matched original-pass controls remain valid. At observed
failed ticks,solver_niter is2 or4,below the unchanged50-iteration cap. Public
Ma force balance and mass*qacc force balance agree closely;the private stopping
criterion is not captured and root cause remains unresolved. All original8
TRAIN rejects stay masked. Prepared common808-TRAIN/324-CHECK/188-pair data for
the next registered four-arm WM fit;zero neural/Actor updates this turn.

## Registered replay and physical outcomes

`pen-train-solver-observation-protocol-v1.md` froze the four R0 failed conditions
and same-source/amplitude/direction opposite-sign valid controls. Separate fresh
processes A/B use three repeats,two recorder scopes and alternating case/scope
order. Each runs48 branches/384 original native steps;total96/768. Every branch
uses full default STRICT source restore;no solver setting,physics,asset,source
repair or installed package changes. No resets or aids during the eight ticks.

PACKET_BASELINE uses the original recorder. PACKET_PLUS_SOLVER_FIELDS adds reads
of existing qacc/qacc_smooth/qvel,smooth/constraint/bias/passive/actuator forces,
solver_niter/nefc,efc.Ma and M after each original post-step packet. It does not
recapture graphs,run forward or assign fields. Sources and observations restore
exactly,actual PD matches the original,and packet vs extra-field epochs agree.

|Fresh process/scope|Failed original condition replays|Matched original-pass replays|
|---|---:|---:|
|A,baseline|9/12 fail|0/12 fail|
|A,extra solver fields|10/12 fail|0/12 fail|
|B,baseline|9/12 fail|0/12 fail|
|B,extra solver fields|11/12 fail|0/12 fail|

Overall39/48 selected-failure executions exceed unchanged5e-7Ns impulse gate;
all48 selected original-pass controls pass. This is deliberate case selection,
not a natural failure rate,probability estimate or task-success comparison.
All other reviewed physical gates pass. Max failing impulse is2.630163e-6Ns;
matched controls max9.92140e-10Ns. Three original failed conditions fail12/12
replays each;RESIDUAL,seed8801,phase36,D1,.005,minus fails3/12. Retain every
outcome,including the9 selected-failure replays that pass;never substitute
these for the original excluded TRAIN records.

## Additional solver evidence and limits

Extra-field failed ticks number21;all have solver_niter2 or4 versus cap50.
Thus these observed failures do not consume the iteration cap. This does NOT
identify which early termination criterion fired or establish a solver bug.
The installed frozen solver source supports multiple termination criteria;
the per-iteration private improvement/decrement/scaled gradient are not exposed
by this protocol. No private context was captured or reconstructed as truth.

Across extra-field records,max norms:

|Recorded comparison|Maximum|
|---|---:|
|mass*qacc-smooth-constraint|1.052070e-3N|
|public efc.Ma-smooth-constraint proxy|1.052070e-3N|
|efc.Ma-mass*qacc|2.51132e-8N|
|smooth force minus gravity|4.58741e-8N|
|delta_v-actual_GPU_dt*qacc|2.10967e-9m/s|

Recorded object passive/actuator forces are zero. These data narrow the dominant
recorded discrepancy to post-solver force balance rather than missing passive
load,a large mass-times-acceleration/Ma mismatch or incorrect EFC relabeling.
They do not prove a private solver termination reason,rule out hidden history
or isolate stream/instrumentation causality. The public Ma proxy is explicitly
distinct from private SolverContext gradient and scaling.

40/48 baseline-vs-extra scope pairs are exactly equal on the five inspected
physical packet arrays;8 differ. Maximum paired component differences across
native records: qpos2.98023e-7,qvel1.68681e-4,qacc.0674725,generalized constraint
force7.62761e-4,and EFC edge force.196708. These arrays mix hand/object DOFs;
qpos/qvel are not one-unit object errors. Do not hide differences or claim
read-only observation guarantees identical future physics. Mode and order are
not independently isolated by this small rotating design. Prior epoch anomaly,
its90 unreproducing replays and original strict8 rejects remain intact.

## Shared physical implementation and checks

New maintained mechanics helper `mechanics/solver_balance.py` performs only
recorded-data algebra with explicit WORLD gravity,mass,free-body DOFs and actual
GPU timestep. Tests distinguish velocity integration,force-balance defect and
Ma mismatch;bad epochs/shapes/nonfinite fields reject. The original runtime,
recorder,source loader,physics,geometry and training implementation are unchanged.

|Evidence|Checks|Actual native work|What passes|
|---|---:|---:|---|
|Fresh process A|266/266|384|Restore/observation/PD/native/field-epoch contract;19 physical failures retained|
|Fresh process B|266/266|384|Same contract;20 physical failures retained|
|Independent saved review|485/485|0|Raw EFC/packet/field/source/options/accounting;not physical feasibility of39 rows|
|Common fit-data preparation|45/45|0|Roles,full Model,masks,scales,pairs and leakage negative control|
|Full unit suite|360/360|0 task native work|Prior352 plus4 solver-balance and4 dataset-contract tests|

## Ready common data;no optimizer yet

Before normalization,`pen-contact-response-fit-protocol-v1.md` and its config
register fresh POINT_DIRECT/POINT_IMPULSE/RESPONSE_DIRECT/RESPONSE_IMPULSE arms.
The fixed next budget is2000 updates/arm,checkpoint100/2000,point256,pairs32,
width512,common initialization and point streams,contrast weight.1. Reuse the
maintained PenResponseWM/update_impulse/update_contrast/checkpoint machinery.
It is equal updates/point samples,not equal FLOPs:response arms add64 endpoints.
The response objective is existing object12+wrench6;Fn6 stays point supervised
and independently response-checked. No silent change of the old objective.

Preparation verifies every available original TRAIN/CHECK source's404 full GPU
Model arrays and scalars (36 sources). Historical frozen snapshots are checked;
no old native state or CHECK labels transfer into TRAIN supervision. Output
keeps816 TRAIN records/mask808 valid,324 NEVER_FIT developmental CHECK records,
188 eligible pair records,original row indices and all192 group audit entries.
The8 rejected rows and4 affected paired groups remain excluded without repair.

Normalization and family-specific head scales use ONLY808 valid TRAIN rows.
Pair targets average the two valid original repeats per sign and retain actual
PD direction/source/warm/dt checks. A deliberate negative control changes CHECK
labels and excluded TRAIN labels by1e8/1e9:all normalizers,head scales and pair
supervision remain exactly equal. Two normalizer computations are accounted
(canonical plus negative-control recomputation),zero neural updates. Unit tests
also reject role mixing,group-mask overrides and wrong actual PD differences.

Common data is `runs/pen_contact_response_common_data_v1/report.json`. It is
ready for the next fit;private solver cause is not a blocker for qualified rows.
The next implementation task is extending the SINGLE maintained fit runner for
this explicit dataset schema,keeping historical frozen fits/receipts untouched.
No CHECK tuning or final-holdout claim;whole-reference splits,multiseed cost,
Actor gradients and genuine ego/hardware validation remain required.

Current64-iteration policy/video is unchanged and still fails the full task.
No ordinary PPO speedup claim follows from these highly instrumented probes.
Driver/CUDA/packages and physical settings are unchanged. Exact sources and
full-path-size-SHA evidence are indexed by `provenance/pen_solver_observation_results`.
