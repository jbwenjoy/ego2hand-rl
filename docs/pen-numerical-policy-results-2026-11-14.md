# Actual device numerical policy and full-case force-quality checks

Previous goal turn progressed ined8a759: fixed action envelopes cover18 cases,
all actual tasks fail, two impulse gates fail, and local TIGHT has mixed coarse
results.166 tests pass. This phase isolates numerical factors, discovers the
host/device tolerance conversion policy, makes an explicit device option and
repeats full original-clock episodes. No neural training or MPC is started.
The full SuperMimic+CHORD Goal remains active.

## Fixed-state factorization does not find a global local-probe winner

`runs/pen_quality_factorial_v1/report.json` restores the SAME two offending
PRE states and original ctrl/qvel/qacc warmstart.128 worlds and three repeats
per variant. All physical parameters and original5e-7N*s impulse gate remain.
Nondefault variants hold main/line-search iteration limits50 while varying
only stopping tolerances; actual device arrays are explicitly set in probes.

|Variant|Main tol / LS tol|5ms passing worlds|2.5ms passing worlds|
|---|---|---:|---:|
|DEFAULT|library defaults|0/384|0/384|
|CAPS50|library defaults|0/384|0/384|
|TOL_ONLY|1e-10 / default|330/384|384/384|
|LS_ONLY|default /1e-6|0/384|0/384|
|BOTH|1e-10 /1e-6|316/384|384/384|
|BOTH_STRICT|1e-12 /1e-8|322/384|384/384|

Coarse worst impulse error remains about6.66e-7N*s. Fractions vary with the
already observed repeated-execution effects and are descriptive, not success
probabilities or evidence that LS tightening worsens the solver. Default/caps
and LS-only stop after2 iterations(coarse) or1(fine); tight main tolerance uses
3–4/2. Main termination tolerance matters here; larger caps alone do not fix it.
No variant passes both source states in all worlds, so preregistered selection
returns NONE. Full execution uses BOTH_STRICT only as a diagnostic fallback.

`runs/pen_quality_factorial_review_v1/report.json`:360/360 independent saved
source/control/warmstart, EFC force/torque, impulse and gate-count checks pass.
The earlier numerical-options source hash is preserved exactly: only an unused
JSON receipt-casting helper changed after this probe. Its explicit source alias
is recorded and checked, rather than ignoring the original hash mismatch.

## Configuring native tolerance did not necessarily configure device tolerance

`runs/pen_quality_episodes_v1/report.json` runs DEFAULT/fallback at both dt
and fixed BASELINE,D4_PLUS,D2_PLUS,D2_MINUS:16 full episodes, first native task
termination, no resets or object aid after start. Requested/native fallback
tolerance is1e-12, but newly recorded actual GPU tolerance is1e-6.

The installed MuJoCo Warp3.11 primary source at
`D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src/io.py:427`
intentionally floors native tolerance to1e-6 during float32 model conversion,
to avoid needlessly oscillating iterations near the solution. This is a library
policy, not a driver/CUDA defect. The previous explicit local probes changed
the device array after conversion; plain environment configuration did not.
Host and device numerical metadata are therefore separate contracts.

All v1 trajectories fail; actual EFC109/112 retains three impulse failures.
v1 is preserved as a configuration-conformance failure, not rewritten as if
the requested GPU tolerance had been active. It does not validate BOTH_STRICT.

## Explicit, opt-in device override with recorded before/after values

New `runtime/numerical_options.py` allows only iterations,ls_iterations,
tolerance,ls_tolerance, with finite positive values and integral iteration
counts. Physical keys such as gravity/mass/integrator are rejected before any
option mutation. The shared evaluator applies native options before environment
construction. Its new numerical_device_override flag defaults false.

When explicitly true, the requested device arrays are set before reset/actual
execution. Capture-time scalar iteration caps must already match; otherwise
execution fails. Existing array pointers are retained, so captured nodes read
updated values. Before/after host/GPU values, policy flag and actual per-tick
solver counts are saved. DEFAULT retains the original library floor. No
installed kernel/library, material/inertia/geometry/gravity/CCD/dt or driver edit.

This intentionally below-floor numerical intervention has no blanket accuracy
or stability guarantee; it is judged from measured force/state receipts.
All source revisions and original outcome metadata remain available.

## Full-case replay with actual device tolerance

`runs/pen_quality_episodes_v2/report.json` repeats the SAME16-case protocol
with explicit device override only for fallback. Device fallback tolerance is
9.99999996e-13 and LS tolerance9.99999994e-9, the requested float32 values.
DEFAULT device tolerance stays9.99999997e-7. Actual iteration counts are bounded
by their declared caps and never exceed8 in this cohort.

|Profile|DEFAULT5/2.5ms duration|Strict-device5/2.5ms duration|
|---|---:|---:|
|BASELINE|0.22 /0.20s|0.22 /0.20s|
|D4_PLUS|0.28 /0.30s|0.28 /0.30s|
|D2_PLUS|0.24 /0.24s|0.24 /0.24s|
|D2_MINUS|0.18 /0.16s|0.18 /0.16s|

All16 actual tasks still fail position/orientation/complete7.36s gates. No
stable manipulation, learned policy improvement, exact determinism, new100ms
fidelity pass, real ego generalization or training acceleration is established.
Equal durations are not byte-identical trajectory equivalence.

Actual EFC/momentum saved review is111/112 overall:

|Numerical arm|EFC checks|Maximum linear impulse residual|
|---|---:|---:|
|DEFAULT8 episodes|55/56|9.84e-7N*s|
|Explicit strict-device8 episodes|56/56|9.24e-9N*s|

The remaining DEFAULT D4_PLUS2.5ms failure is retained. The strict-device
cases pass the SAME original force1e-4N,torque1e-5Nm and impulse5e-7N*s gates,
including all actual recorded prefixes. This is evidence for numerical label
quality on this fixed case set, not a global repair: the prior isolated coarse
probe still has failures, and full task tracking remains failed.

`runs/pen_quality_episodes_review_v2/report.json`:240/240 own checks pass:
requested/native/device policy, float32 values, initial selected pose/source
velocity, original raw actions/EMA/PD/phase, state carries, no-aid, first
terminal, metrics and unchanged task gates. All15 physical tables are equal
across16 cases. Actual source-to-target/control and metric reconstruction
errors pass unchanged limits. Combined review remains FALSE because DEFAULT
has one failed impulse check; its command intentionally exits nonzero.

169 repository tests pass, including rejecting physical keys before mutation,
invalid iteration-limit coercion and preserving tight requested precision.
These are code checks, not169 successful physical tasks. Earlier archived
studies, external environments, checkpoints and installed simulator are intact.

## Reproduction and next controlled-learning work

Use the existing external Hand2 Python from the unified repository:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/probe_pen_numerical_quality.py --config configs/pen_quality_factorial_v1.toml --output runs/FRESH_FACTORIAL
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/run_pen_quality_episodes.py --config configs/pen_quality_episodes_v2.toml --output runs/FRESH_EPISODES
```

Configs pin source reports. Fresh upstream artifacts need a new config; frozen
inputs are not overwritten. Saved reviewers accept explicit verified source
aliases only where needed. Logs, model/trace/settings pins and negative
outcomes are recorded in provenance; large numeric assets remain ignored.

Next acquire FRESH explicitly TRAIN/CHECK controlled transitions under a
versioned actual-device numerical policy and unchanged force-quality gates.
Do not repurpose current CHECK/diagnostic nonexpert records as experts. The
object-first joint indexing, source-BODY mean wrench, all five fingers/palm
contact channels, actual PD targets and original EMA/phase must be explicit
before reusing the maintained state/impulse world-model code. Validate real
action-response uncertainty and gradients rather than requiring a fictitious
byte-exact branch. WM fitting on genuine failure transitions is a dynamics
question, distinct from expert imitation or successful actor qualification.

No neural optimizer/checkpoint was started this phase. The first healthy
future neural checkpoint still triggers restore verification, ETA and requested
Goal pause. Original matched A1/A2/support/WM comparisons, complete no-aid tasks,
independent ego RGB, shared multi-trajectory total costs and hardware contracts
all remain required.
