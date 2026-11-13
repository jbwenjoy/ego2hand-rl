# Buffer diagnosis, bounded actions and numerical force-balance quality

Previous goal turn made progress in4a70400: exact404-array/195-scalar model
restoration, failed100ms step fidelity,192/192 saved review and163 tests.
This phase isolates fresh/reused buffers, collects prospective bounded-action
predictions and actual trajectories, and probes the two new momentum failures.
No MPC, neural optimizer, driver/CUDA or installed engine changes occur.
The full Goal remains active; all experiments remain controlled upstream pen,
not real ego input, learned Actor success or a matched SuperMimic A1/A2 trial.

## Fresh data does not eliminate the observed100ms spread

`runs/pen_buffer_diagnosis_v2/report.json` follows v0.66:both dt,128 worlds,
three repetitions, new-versus-reused clone allocations. Time0 executes100ms;
20/40ms anchors take one physical step from exact saved qpos/qvel/ctrl/time
and last-step qacc warmstart. All216/216 input/control/capacity/no-aid/finite
checks pass.100ms fidelity stays24/72, unchanged and NOT ready for planning.

| Same-input spread from world0 over100ms |Reuse5ms|Fresh5ms|Reuse2.5ms|Fresh2.5ms|
|---|---:|---:|---:|---:|
|Object position|3.020mm|3.162mm|1.648mm|1.648mm|
|Object rotation|0.15832rad|0.16557rad|0.11938rad|0.11938rad|

Fresh means a new clone allocation under the maintained constructor, which
does one explicitly counted setup integration and reset. It is not pristine
untouched put_data memory. Results do not exclude every hidden workspace
effect; they show that reallocating through this path is not sufficient.

New `mechanics/constraint_alignment.py` matches actual source rows offline by
contact geoms/geomcollisionid/pyramid edge or noncontact type/id. Duplicate,
ambiguous and invalid addresses fail. It NEVER reorders a live solver buffer.
Raw source addresses, dense J,D,aref/force, contact frames/points, smooth
forces/accelerations and iteration counts are preserved for first60ms.

At each exact-input anchor, all128 worlds have matching physical contact and
constraint identities. Source row layouts vary (up to50 orders). After
alignment, maximum first-step differences across the declared probes are:
J1.49e-8,D0,aref2.38e-7 and EFC force2.09e-7N. Smooth force/acceleration also
have tiny differences in some probes. Numerical differences therefore exist
before accumulated state drift; permutation alone is not a complete equality
test. No solver iteration cap is reached in these recorded probes. These
facts do not prove a unique kernel cause or rule out other convergence issues.

## Prospective action envelopes are a separate prediction question

v0.67/config preregisters baseline and +/-0.04 raw-action offsets in four sine
directions, applied for the first five20ms intervals and then removed. Original
scale/EMA/target limits, selected initial pose, velocities, object/wrist,
gravity/materials and both dt remain. The additive action-provider callback
uses the same shared evaluator, first native termination and no post-start
resets. It creates no Actor and preserves the old evaluator bytes/source alias.

`runs/pen_action_envelope_v1/report.json` contains all18 cases. Baseline and
direction1+/- are CALIBRATION(6); directions2..4+/- are CHECK(12). All are
NONEXPERT_NOT_FIT; the observed best profile does not change those roles.
Before each actual execution,128 identical-input prediction worlds run the
known future100ms command sequence with the same readonly live device model.

At every POST tick, the actual position and WORLD rotation-log vector relative
to the original reference are compared with sample min/max bounds plus fixed
0.5mm and0.01rad COMPONENT margins. CALIBRATION6/6 and CHECK12/12 have complete
100ms coverage under these fixed margins. Two cases have95% raw coverage;
others100%. Raw zero-width rotation endpoints are subject to floating roundoff.
No data-dependent margin inflation, seed/profile exclusion or shorter horizon.

This is a sample envelope with a deterministic margin, NOT a calibrated
probability, confidence guarantee, exact replay, new planning-readiness claim
or real-data/general-action validation. Some rotation spans reach0.1676rad and
position spans3.832mm; broad coverage cannot establish accurate narrow prediction
or task success. The original100ms deterministic fidelity gates remain FAILED.

All18 actual task trajectories fail. Recorded durations:

|Profile|5ms|2.5ms|
|---|---:|---:|
|Baseline|0.20s|0.20s|
|D1+ / D1-|0.26 /0.18s|0.26 /0.18s|
|D2+ / D2-|0.24 /0.18s|0.26 /0.16s|
|D3+ / D3-|0.24 /0.18s|0.24 /0.18s|
|D4+ / D4-|0.30 /0.18s|0.30 /0.18s|

The0.30s profile is one prespecified CHECK intervention, not a selected expert
or policy improvement verified on a fresh test. No complete7.36s trajectory,
stable manipulation or learned control benefit is demonstrated.

## Two full-prefix impulse checks fail despite envelope coverage

`runs/pen_action_envelope_review_v2/report.json` independently passes306/306
action/role, EMA/PD, frame/log-envelope, fixed-margin, capacity/no-aid and
first-terminal checks. Its existing actual EFC subreview is124/126, so
`combined_review_passed=false` and the review command intentionally exits
nonzero. The force/torque EFC identities and label decode pass for every case;
two original LINEAR IMPULSE checks fail, both D4_PLUS:

|dt|Failed POST time|Impulse residual|Original threshold|
|---|---:|---:|---:|
|5ms|0.120s|6.65e-7N*s|5e-7N*s|
|2.5ms|0.055s|9.84e-7N*s|5e-7N*s|

The second failure lies inside the covered100ms forecast. Coverage of the
same approximate plant is therefore not a substitute for physical balance
quality. No failed receipt is silently dropped, threshold relaxed or case
promoted to training supervision. The failed v1 review/EFC report is preserved;
v2 consumes that same EFC report and completes the independent other checks.

`runs/pen_action_force_balance_v1/report.json` decomposes all18 saved runs:
free-body M*a+bias-passive-actuation-applied-EFC_wrench, integration roundoff,
and gravity/bias precision correction. The impulse identity is reconstructed
to4.20e-19N*s;54/54 algebraic checks pass. At offending steps, integration
roundoff is of order1e-11N*s, while the final linear equation residual reaches
1.33e-4/3.94e-4N. Maximum integration roundoff across entire prefixes is below
6.40e-10N*s. Final force-balance error dominates the two impulse failures;
this does not by itself identify a sole solver stopping/kernel defect.

## Raising the iteration cap alone is not a repair

The explicit post-failure v0.68 diagnostic restores each EXACT offending PRE
state, source ctrl and preceding qacc warmstart. Native topology and all15
physical fields are checked against the404-array model source before exact
restoration. One physical step is repeated three times in128 worlds for each
declared numerical variant, preserving geometry/material/inertia/gravity/CCD/
integrator/dt. Only numerical stopping parameters change:

|Numerical variant|5ms passing worlds|2.5ms passing worlds|
|---|---:|---:|
|DEFAULT10 iterations/default tolerances|0/384|0/384|
|ITER50:only iteration limit50|0/384|0/384|
|TIGHT:50 iterations,50 LS, tol1e-10, LS tol1e-6|324/384|384/384|

`runs/pen_numerical_quality_probe_v1/report.json`: DEFAULT/ITER50 actually
stop after2 iterations(coarse) or1(fine). TIGHT uses3–4(coarse) and2(fine).
Fine worst residual falls to2.48e-10N*s. Coarse retains60 failures, with worst
6.66e-7N*s. TIGHT is NOT an accepted global repair or a qualified physical
trajectory. Original DEFAULT runs remain unchanged/failed. A simultaneous
change of tolerance and line-search settings does not isolate their effects.

`runs/pen_numerical_quality_review_v1/report.json` independently passes180/180
source-state/warmstart/ctrl, raw EFC decode, full COM force/torque, impulse and
gate-count readbacks. These verify mixed outcomes; they do not turn all worlds
green.166 repository tests pass, including physical-row permutation identity,
invalid address rejection and rejecting ambiguous controllers before creating
an episode. Historical archives, environments and installed engine are intact.

## Reproduction and next work

Use the existing external Hand2 Python from this repository:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/diagnose_pen_buffers.py --config configs/pen_buffer_diagnosis_v1.toml --output runs/FRESH_BUFFERS
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/collect_pen_action_envelopes.py --config configs/pen_action_envelope_v1.toml --output runs/FRESH_INTERVENTIONS
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/probe_pen_numerical_quality.py --config configs/pen_numerical_quality_probe_v1.toml --output runs/FRESH_QUALITY
```

Configs pin existing source reports; fresh upstream outputs require a new
explicit config, not rewriting frozen runs. Review scripts retain failure
exit codes. Artifacts/provenance preserve exact source aliases and all raw
failures, including an early JSON-metadata development error.

Next isolate numerical stopping conditions and confirm any proposed repair
on fresh full original-clock cases and their actual force/impulse receipts.
Use calibrated action-response uncertainty when designing moving-contact
control/WM gradient checks; current envelope coverage alone cannot license
that claim. Current CHECK/diagnostic data remain nonexpert/not-fit. Fresh
training data need explicit roles and balance-quality masks before neural
learning. No neural checkpoint was produced here; the first healthy future
checkpoint still triggers restore verification, ETA and requested Goal pause.
All original controlled learning, matched A1/A2/support/WM gradients, ego,
multi-trajectory total costs and hardware contracts remain required.
