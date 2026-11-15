# Independent-noise TRAIN response coverage — 2026-11-16

Collected the registered24 source states and816 branches using default STRICT
restoration, with8800 native integrations and zero optimizer/Actor updates.
All sources and branches pass online geometry/control checks. Independent EFC,
clock,frame,PD,warmstart and momentum preparation retains808 valid rows and
rejects8 impulse-inconsistent rows. Existing324 developmental CHECK rows remain
NEVER_FIT. No new WM fit, improved policy/video or ego result is claimed.

## Data and input contracts

The maintained collector now supports an explicit schema2 TRAIN configuration;
the historical schema1 CHECK path keeps its role, phases and27-probe protocol.
Shared physical capture,restore,chart and decoding remain in `src/`. Four new
W1 executions use the two frozen final64 actors and PCG64 raw-noise seeds8801/
8802,sigma.01. Each runs71 controls continuously with no post-start reset. The
complete2272-native source bank finishes before any counterfactual restoration.
All prefix controls retain actor mean,submitted raw action and clipping count;
none clipped. Paired seeds between arms are not independent trained seeds.

Actor/normalizer weights transfer through the explicit frozen-weights helper;
no old native,optimizer,critic or RNG transfers. All404 W1 Model arrays and
scalar fields match historical W1 context. CHECK context is used only for Model
compatibility,not as new source states or response supervision. Source phases
12/24/36/48/64/72 correspond to.22/.46/.70/.94/1.26/1.42s of the SAME original
reference. All12 paired same-arm/phase comparisons have distinct physical states
under the two noise histories. Max component qpos differences range.00270–.03080
(mixed joint/position/quaternion coordinates;not a single physical-unit error).

Each state has34 branches:2 baseline repeats and4 fixed random unit directions
(PCG649301),2 actual-PD amplitudes(.002/.005rad),2 signs,2 repeats. The4x raw
perturbation passes unchanged clamp/scale/EMA/PD and is checked against actual
held target20 within2e-7rad. No clipped perturbations or source replacements.
Each branch executes eight2.5ms ticks;probe work is6528 native integrations.
All source/observation/full-native restore audits pass. Epoch mode is disabled.

## Independent preparation and retained physical failures

The maintained decoder uses actual GPU EFC edges;the record packer uses actual
engine warmstart,held PD and source-qpos BODY axes about moving COM. Chart remains
state55+engineWarm26+actualPD20+nativeDt1 -> delta52/nextWarm26/wrench6/Fn6.
Original geometry2mm/.2mm/.001rad and all force/torque/impulse/no-aid/overflow/
nonhand/finite masks remain. Rejects stay in arrays and metadata,with valid=False.

|Evidence|Observed result|Scope|
|---|---:|---|
|Collection contract|5/5|24 sources,816 available online-qualified branches,8800 actual native steps|
|Preparation contract|81/81|Noise history,profile/clock/source/warm/PD provenance and accounting|
|Physical/label row gates|16312/16320|Eight failures,all EFC linear impulse;808/816 rows retained valid|
|Paired response availability|188/192|Every repeat and both signs must be valid;not a smooth-gradient qualification|
|Full unit suite|352/352|Includes deterministic schedules and CHECK mixing/relabel/epoch rejection|

Maximum generalized-force discrepancy is4.77333e-7N,torque8.17571e-9Nm. Maximum
linear impulse discrepancy is2.63016e-6Ns,above unchanged5e-7Ns gate. Rejected
row indices are72,88,137,155,273,291,790,806. All other independent row gates pass.

|Original same-condition repeat pair|Max impulse discrepancy(Ns)|Qacc force-balance discrepancy(N)|
|---|---:|---:|
|RESIDUAL,8801,phase36,D1,.005,minus|2.630163e-6|1.052058e-3|
|RESIDUAL,8801,phase64,D1,.002,plus|1.799760e-6|7.199389e-4|
|RESIDUAL,8802,phase36,D1,.002,plus|2.598552e-6|1.039433e-3|
|SUPPORT,8802,phase72,D2,.005,minus|1.065314e-6|4.260527e-4|

Both original repeats of each condition have identical recorded residuals.
This is a same-process observation,NOT a fresh-process reproducibility claim.
STRICT-path examples establish that the phenomenon can occur without a source
validation epoch. They do not explain the prior epoch counterexample or rule
out any stream,history,solver or instrumentation mechanism.

## Offline momentum decomposition;root cause unresolved

On all816 records,compare delta_v to dt*qacc and decompose the original decoded
impulse into integration residual, qacc force balance and EFC force discrepancy.
Across all records,max velocity-update residual is3.87949e-9m/s. At failing ticks
the dominant term is dt*(mass*qacc-qfrc_constraint-mass*gravity),not EFC relabeling
or a large velocity-integration discrepancy. Recorded object linear damping,
armature,COM offset,gravity compensation and fluid density/viscosity are zero.

The first offline decomposition forgot the decoder's host-vs-GPU gravity
representation and native-dtype multiplication;its9/825 failed identity checks
and report remain immutable. The maintained corrected diagnostic includes those
explicit bookkeeping terms,without changing the original decoder,data mask,
physical threshold or any dynamics. Its825/825 checks verify the decomposition
identity(max error8.70e-20Ns) and accounting,NOT physical feasibility of the8 rows.
The original8 rejects remain unchanged. No new native integration or fit occurs.

Recorded host Model uses Newton,50 iterations,tolerance1e-12,line-search50 and
1e-8 tolerance. Frozen GPU Model inspection confirms50/50 and their float32
tolerance representations,graph_conditional=True. Merely inspecting options
does not prove convergence or attribute a root cause. Next use targeted,
read-only observations of solver iteration/smooth force/M*qacc/gradient at the
retained cases and matched passing controls;retain original history and monitor
whether observation changes reproduction. Do not tune settings to erase a gate.

## Scope and continuation

No scales were fitted and no model or policy was trained. This collection adds
contact-rich TRAIN physical responses for the next registered matched point/
response fit. It does not turn same-reference developmental CHECK into a final
test,demonstrate unseen-reference/ego transfer,qualify a full20D Jacobian or
establish PPO speedup. Prior WM action gradients remain failed and cannot drive
Actor optimization. A new fit must use common initialization/data/updates and
valid-TRAIN-only scales,then independent forward/contact/finite-response/JVP
checks. Whole-original-trajectory splits,multiple trained seeds and total-cost
experiments remain required for generalization/amortization conclusions.

Raw reports live under `runs/pen_response_train_coverage_v1`,
`runs/pen_response_train_preparation_v1`,and both
`runs/pen_response_train_momentum_diagnosis_v1`(failed identity bookkeeping) and
`v2`(corrected algebra). Exact source snapshots,full-path-size-SHA aliases and
prior-chain validation are indexed by `provenance/pen_train_coverage_results`.

Ordinary PPO geometry/synchronization profiling remains separate. No installed
packages,driver or CUDA edits;no physical settings changed. The saved64-iteration
video remains the current policy demonstration;full7.36s controlled task,
real ego assets/frames/dynamics and hardware deployment are still incomplete.
