# Velocity cost does not improve control; a visited state fails20ms prediction

Previous goal turn was PROGRESS: six feedback-support cases, complete saved
reviews and all-candidate analysis changed the next action to plan v0.77.
This turn executes its eight-case velocity comparison and retains failures.
Full Goal remains ACTIVE/incomplete. No Actor/network update, successful
expert, stable pen task, surrogate acceleration or real ego claim is made.

## Shared code and immutable source evidence

Before editing the four pinned shared files, their exact preceding bytes
were copied to `archive/native_pen_pre_velocity_v1`. Explicit original
path+size+SHA to archived pin mappings are recorded in
`provenance/native_pen_pre_velocity_v1/relocations.json`. Old reports,
configuration, arrays and previous archives remain unchanged. Shared planner,
scorer, experiment and review are maintained once in `src`; two new generic
entry points call them. Old drivers stay historical.

Collector83a2f6f and `configs/pen_native_velocity_v1.toml` were committed before
execution. The added endpoint WORLD velocity cost is
`sum(((v_end-v_ref[k+1])/0.25m/s)^2)`, weight1, with original derivative-valid
tail masks. Scale is5mm/20ms, fixed before execution. Original STATE_ONLY and
STATE_PLUS_SUPPORT controls and both velocity variants share the same pose,
model, original7.36s clock, seed8101/57 candidates,20ms horizon, raw/EMA/PD
mapping, support weight and physical/task/replay gates. No new feedback gain,
object force, post-start live reset or physical parameter adjustment occurs.

Each new collection captures237 source files as immutable bytes under its
own ignored run directory, with explicit observed active paths, code revision
and before/after byte verification. This is a source superset, not evidence
that every captured module executed. Later active-code repairs do not rewrite
the source used for these cases. The review also captures its own source
bytes separately from collector code.

## Collection failure and explicit read-only recovery

All eight episodes finished and their per-case reports,134 source packets,
candidate forecasts and `progress.json` were saved in
`runs/pen_native_velocity_v1`. Final root-report assembly then failed because
the refactor removed `sys` while retaining a final `sys.modules` enumeration.
This is an implementation error, not a simulator failure or observation
timeout. The original process is terminal exit1 and original files stay intact.

`tools/recover_pen_native_report.py` verifies the exact recorded final error,
all eight predeclared cases against their individual reports and source/data
pins, and live source-snapshot equality BEFORE repair. It writes a separate
`runs/pen_native_velocity_recovery_v1/report.json`, SHA256
`e7e53793d9bb90d0cbe5d67aa4d0cb586316ada66b5e5b6f7dd295c3b813d34b`.
Recovery adds zero dynamics and zero neural updates. It explicitly records
that the original process's executed-task-module enumeration was never saved:
`task_sources=[]`, `source_import_enumeration_available=false`. A fresh process
is not used to fabricate the lost list. Source-provenance qualification stays
FALSE even though the captured source superset and actual data are intact.

Commit78f3332 moves live import enumeration into a tested helper for future
runs. Unit tests exercise a live namespace, external/untracked imports and
source drift. No physical experiment was rerun just to replace this failure
with a cleaner report. The completed traces remain negative development data.

An initial historical-source review also found an inner contact reviewer
still verifying old active paths despite outer relocation checks. That failed
setup/log is preserved. The repair builds an explicit new read-only review
view with exact archived source paths and original-report/mapping pins, then
passes it to the inner reviewer. Actual case data and physics are unchanged.
The fresh historical review reproduces the original1685/1686 contracts,
41/42 EFC and101/102 selected fidelity, including the same failures.

## All actual tasks fail; velocity additions do not help survival

|dt|objective|duration|COM RMS / max|
|---|---|---:|---:|
|5ms|STATE_ONLY|0.42s|18.21 / 55.13mm|
|5ms|STATE_PLUS_SUPPORT|0.36s|22.67 / 65.26mm|
|5ms|STATE_PLUS_VELOCITY|0.30s|24.91 / 65.93mm|
|5ms|STATE_PLUS_VELOCITY_SUPPORT|0.34s|20.27 / 55.24mm|
|2.5ms|STATE_ONLY|0.34s|19.69 / 57.97mm|
|2.5ms|STATE_PLUS_SUPPORT|0.30s|18.21 / 51.74mm|
|2.5ms|STATE_PLUS_VELOCITY|0.32s|18.81 / 54.66mm|
|2.5ms|STATE_PLUS_VELOCITY_SUPPORT|0.30s|20.33 / 57.04mm|

All terminate for native COM error over50mm. Every unified COM/orientation/
duration gate fails; full-orientation RMS is60.6–67.7deg (directed shaft cost
does not constrain axial roll). Geometry/self/joint/nonhand/no-aid/finite/
capacity gates pass, maximum penetration about1.977mm, zero fallbacks and
unchanged full planning models. All prefixes remain nonexpert.

Velocity state-only lasts less than state-only at both dt; support+velocity
is worse/equal to support alone on duration. Same-seed control trajectories
also vary across prior cohorts. These are not independent statistical trials,
causal millisecond gain estimates or equal-total-compute learned-policy A1/A2.
The same57 candidates/decision isolate the declared score change locally;
unequal episode lengths imply different total compute.

Total work:134 decisions,7638 candidate20ms rollouts,44916 candidate physics
ticks +788 actual ticks =45704 integrations. Recovery/reviews/diagnostics add
no integrations. Common prefix is15 controls/0.30s for all four modes at each
dt. Mean candidate CUDA cost is133–141ms at5ms dt,257–263ms at2.5ms; wall cost
including artifacts171–179ms or299–304ms per decision. No real-time or
GPU-utilization/speedup conclusion follows from these timings.

Common-prefix mean endpoint COM errors (state/support/velocity/velocity+
support) are3.87/7.82/17.75/9.68mm at5ms and8.56/12.66/10.47/14.34mm at2.5ms.
Thus offline improved endpoint-velocity choices from the prior diagnostic
did not demonstrate improved executed control. No weight sweep follows.

## Independent review retains substantial forecast failure

`runs/pen_native_velocity_review_v1/report.json`:2752/2752 saved numerical/
control/source-state/score/target/mask/argmin contracts and actual EFC56/56
pass. Source-provenance qualification is separately FALSE because of the
lost import enumeration; these counts are not a global qualification.
Passing actual EFC for this cohort does not repair the earlier41/42 failure.

Selected forecast versus subsequent actual20ms execution passes all original
state/control/time/force gates at132/134 decisions. Both failed comparisons
remain failed; the review saves results and exits1.

|condition|object rotation error|velocity error|WORLD force / torque error|
|---|---:|---:|---:|
|5ms velocity+support, k9/source180ms|0.038332rad|0.077469m/s|0.464814N / 0.0148497Nm|
|2.5ms state-only, k1/source20ms|0.0000153rad|0.0000273m/s|0.000327N / 0.0000527Nm|

The first also fails WORLD angular velocity (7.66653rad/s); COM error0.387mm
remains below its0.5mm replay gate. Its selected PD targets and time match
exactly, source q/v/warmstart/clock and live isolation checks pass. Numerical
cost checks do not qualify an inaccurate state/force forecast.

`runs/pen_native_velocity_fidelity_ticks_v1/report.json` includes EVERY failed
selected forecast at every physics tick, with unchanged gates. At source180ms,
the first three5ms ticks pass; force discrepancies are at most1.51e-7N and
rotation at most5.65e-8rad. At the fourth tick (20ms), force/state diverge to the
large errors above. At the fine-dt source, first five ticks pass; force/torque
begin failing at tick6 (15ms elapsed). Timing alone does not prove a contact/
active-set or solver cause. Predicted contact rows and solver iterations were
not saved and are not invented for that comparison.

The earlier calibrated sources only extended to120ms; they never qualified
arbitrary visited sources or longer sequences. Before extending the horizon,
next investigate repeatability at these newly failed sources versus matched
passing controls, with complete model/Data/solver/PD evidence. See v0.78.

## Validation, limits and full objective

202 tests passed before collection;203 pass after the final-enumeration
repair. Added tests cover explicit free-object velocity indexing, geometry
precedence, zero-weight/invalid-tail bitwise equivalence to archived scoring,
immutable byte identity/explicit relocation, and final live-module discovery.
Tests isolate native work on CPU; actual RTX executions are recorded
separately. No drivers, CUDA, installed libraries or historical environments
were changed. The new unit does not retroactively supply the lost import list.

Model404 arrays/195 scalars and source169 dynamic arrays are saved. Dynamic
Data scalar leaves were checked in memory against the live source but not
serialized in this native-controller run. Future reconstruction must identify
its static-capacity metadata as reconstructed, not original observations;
fresh diagnostic captures must save those scalars and full layout explicitly.
This limitation does not erase recorded actual results or justify loose replay
gates. Compact evidence, test logs and explicit source mappings are retained
in `provenance/pen_native_velocity_results`; large raw packets remain ignored.

The original scope still requires learned SuperMimic/CHORD control/save-resume,
fair support-policy comparisons, contact/gradient verification, cube/pen
unified evaluation, shared multi-trajectory total cost, true ego input/frame/
asset/dynamics evidence and hardware zero/gain/latency/observation contracts.
Neither nominal physical priors nor privileged simulator state replace ego
or hardware observations. No stable task or fresh Actor update occurred.
All current handles are terminal; no training/checkpoint wait. Goal ACTIVE.
