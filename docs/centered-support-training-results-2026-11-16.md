# Centered support score and64-iteration controlled trial — 2026-11-16

Both registered arms completed64 iterations,131072 transitions,1048576 native
world-integrations and2048 Adam steps EACH. The centered-support actor executes
2.14s in three fresh W1 numerical repeats; residual executes1.86/2.08/1.86s.
All nine executions, including initialization, fail the original7.36s task.
Support trades somewhat lower common-horizon COM error for higher orientation
RMS and eventually violates self-collision. This is mixed single-seed/clip
evidence, not established support benefit, ego generalization or WM gradients.

## Offline definition selected before training

`support-score-study-protocol-v1.md` and its config/source receipt predeclare
ABSOLUTE,CONTACT_ABSOLUTE,CENTERED,CONTACT_CENTERED on identical saved pilot rows.
CENTERED alone satisfies every chosen analytic invariant: zero at zero NET
wrench even for a balanced squeeze, negative opposite-target score, exact
nonzero target uniquely best, and no jump at a.1N contact-load threshold.
Expected counterexamples for other variants remain in the report. Four saved
rollouts independently match NumPy scores. Analysis14/14,zero new physics/updates.

The selected term is exp(-L(measured,target)/2)-exp(-L(0,target)/2), with the
original force.3N and torque.003Nm scales and original eligibility. It is signed,
bounded[-1,1] and does not add a contact threshold. Centering depends on phase
and is NOT potential-based policy-invariant shaping. It can change termination
incentives and does not bound squeezing,prove contact feasibility or provide
physical action gradients. Geometry/no-aid/full tracking gates remain unchanged.

## Maintained code and explicit schema

One maintained `SupportPenRuntime`/`SupportPenPPO`/collector implements both
declared score variants; experiment differences remain in configs. The helper
`mechanics/support_score.py` separates wrench scoring from physical feasibility.
Old absolute configs retain their formula; production permits explicit ABSOLUTE
or CENTERED only. New CENTERED requires public schema
`pen_geometry_support_PPO_training:4`; old training3 cross-loading rejects.
Forks require the same explicit schema and may change only arm/support scalar.

Historical full-native states require their exact frozen source bytes. Editing
maintained files is not an implicit historical continuation/migration. Prior
snapshots/full-path-size-SHA aliases remain authoritative; this trial constructs
a fresh shared source, never imports old trained weights. Runtime geometry,
source axes/MOVING COM,USED k->interval[k-1,k],365 valid targets,last3 invalid,
FP64 canonical reference,161/188 observations,total raw clamp/scale/EMA/PD,
compound asset,mass/inertia/friction/CCD/solver and original clock stay unchanged.

`pen-centered-support-training-protocol-v1.md` froze the new budget BEFORE
optimization. Both arms share freshly calibrated modules,normalizers,Adam/native
state and RNG; online trajectories may diverge. Each has64x64x32 controls and
2048 Adam steps. Full checkpoints save at1,8,16,...,64 with separate memory;
every iteration saves25/25 collector health. Unsaved iterations are not called
disk checkpoints. First checkpoint bytes/counters/health pass; ETA was4–6min.
Measured collection sums102.21/102.83s,optimizer7.65/7.48s for residual/support;
these exclude some construction/cold-check/save costs and are not speedup data.

## Qualification, recovery and actual evaluation

| Evidence | Result | What is verified |
|---|---:|---|
|GPU qualification|75/75|Selected score,shared source/fork,mean targets/frames,reward/phase,and full reload;4096 native,0 updates|
|Registered training|5/5|Equal declared budgets;all128 collector blocks25/25|
|Fresh complete-state/actual continuation|216/216|Native/cache/task/network/Adam/norm/memory/counters,atomic rejection of training2/3,wrong arm,corrupt target; actual4-control update/reload per arm|
|Independent nine-execution review|351/351|Raw active EFC/contact,poses,fullSO3,geometry,signed WORLD/source moving-COM wrench,loss AND declared centered score|
|Common-horizon coverage|15/15|All repeats retained;post-hoc descriptive windows,no new dynamics|
|Full unit suite|334/334|Five new zero-net/torque/gate/schema counterexamples plus prior329|

Evaluation repetition/order/video choice was additionally registered BEFORE
execution in `runs/pen_centered_support_evaluation_registration_v1`. Each of
INITIAL,RESIDUAL,SUPPORT is executed fresh three times in that order. Only
actor/norm transfers from W32 to W1; no native-state transfer,action sampling,
post-start reset or aids. Video uses repeat0,never the best repeat. Repeats
quantify numerical variation of a fixed actor,NOT independent trained seeds.

|Case|Durations(s)|Full-task passes|First run stop|
|---|---|---:|---|
|INITIAL|.20/.20/.20|0/3|Position error>50mm|
|RESIDUAL|1.86/2.08/1.86|0/3|Object depth>2mm|
|SUPPORT|2.14/2.14/2.14|0/3|Self depth>.2mm|

Repeat0 full-duration COM RMS/angle RMS are residual9.900mm/19.930deg versus
support10.737mm/30.057deg. These durations differ. The independent post-hoc
descriptive comparison therefore also keeps ALL_COMMON=.20s(min of all9) and
FINAL_COMMON=1.86s(min of the6 trained-actor runs), with every repeat included:

|Same first1.86s,median across numerical repeats|Residual|Support|
|---|---:|---:|
|COM RMS|9.900mm|7.952mm|
|COM max|40.934mm|21.359mm|
|FullSO3 RMS|20.411deg|22.313deg|
|FullSO3 max|81.356deg|55.909deg|
|Object depth max|2.140mm|1.979mm|
|Self depth max|.068mm|.063mm|

Both violate original tracking gates. Support's orientation RMS is worse even
when its orientation MAX and COM errors are lower; duration alone is insufficient
to declare a better controller. Residual first exceeds full-angle10deg at
.650–.6575s; support.710–.7125s. First COM10mm breaches are1.555–1.5575s and
1.545–1.5525s respectively. Self-collision breaches occur at2.135–2.1375s for
support. Control-boundary termination can follow a native breach by up to the
remaining control interval; no claim of mid-native stopping is made.

At the first.20s both trained actors satisfy the instantaneous error/geometry
thresholds (median COM RMS1.010/.943mm,full-angle RMS3.504/2.667deg), but this
short legal prefix is not substituted for the full task. Training samples also
contain larger control-window penetration/self violations; do not call all
sampled data legal because most transitions precede failure.

## Reproduction and cost

Roots: `pen_support_score_study_v1`, `pen_centered_support_qualification_v1`,
`pen_centered_support_training_v1`, `pen_centered_support_state_review_v1`,
`pen_centered_support_evaluation_registration_v1`,
`pen_centered_support_evaluation_v1`, `pen_centered_support_evaluation_review_v1`,
`pen_centered_support_common_windows_v1` under `runs/`.

Commands use tools `compare_pen_support_scores.py`, `qualify_pen_support_pilot.py`,
`train_pen_support_pilot.py`, `evaluate_pen_support_repeats.py`,
`review_pen_support_pilot.py`, `review_pen_support_preview.py`,
`compare_pen_support_executions.py` with the corresponding registered configs
and fresh output directories. Reports pin all concrete inputs and snapshots.
Latest video is `runs/pen_centered_support_evaluation_v1/geometry_support_pilot_comparison.mp4`.
It was decoded and visually inspected;20x slow,actual saved2.5ms poses,no
interpolation,explicit terminal stills. It remains controlled perfect-pose data.

This cohort costs2108424 native world-integrations:4096 qualification +2097152
registered training +5128 all9 evaluations +2048 diagnostic loaded continuation.
Registered updates4096 plus64 diagnostic updates,total4160 actual Adam steps.
The latter are not pilot candidates. Score analysis,registration,independent
readback,matched-window calculations,rendering and unit tests add0 native/Adam.
Keep previous cohorts/costs separate; no installed package/driver/CUDA changes.

Full original objective remains ACTIVE/incomplete. Next work must address
contact-feasible torque/orientation control and physical WM gradients on
qualified data; further epochs alone do not prove feasibility. Preserve original
cube/pen full-task gates,multitrajectory/multiseed A1/A2,source/noise issues,
real ego frame/geometry/dynamics and hardware observation/subversion contracts.
