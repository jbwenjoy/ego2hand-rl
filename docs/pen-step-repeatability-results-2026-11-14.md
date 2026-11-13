# GPU model snapshots and the failed100ms step fidelity gate

The previous goal turn made progress in d093fbc: bounded first-frame support
passes but actual constant-residual trajectories fail at0.22/0.20s. This phase
implements physical STEP clones, exact device-model snapshots and independent
repeatability checks. It does NOT establish100ms prediction fidelity or start
MPC/neural training. The full SuperMimic+CHORD Goal remains active.

## Original clock and controller are preserved

New `runtime/gpu_pen_rollout.py` reuses the resident forward oracle and adds
captured physical step, initial qpos/qvel/ctrl/warmstart/time/EMA restoration,
original phase1..5,0.5 residual scale,[-0.6,0.6] residual clip,alpha0.5 and
original legal PD-target clipping. The model rejects activation/history,
plugins, sleep and hidden actuator dynamics. Fixed mocap/no-aid contracts
remain. Boundary forward refresh matches the task after every20ms interval.
Each clone is independent of actual execution; only planning worlds rewind.

The intended planner horizon is100ms. v0.64 declares six fidelity gates:
max hand q2mrad, object translation0.5mm, rotation0.01rad, hand velocity
0.2rad/s, object linear velocity0.01m/s and world angular velocity0.2rad/s.
These thresholds are unchanged. Full failed-prefix errors are reported
separately from the100ms check; no short-horizon passing result replaces it.

`runs/pen_gpu_step_audit_v1/report.json` uses the old declared15 physical
tables and saved host topology, single/batch128, both5/2.5ms. It passes56/72:
initial state, controls/EMA, clocks, capacity, no-aid and EFC force/moment
identities pass, but all four object fidelity checks fail in every cohort.
The six100ms dynamics gates are a separate question from EFC force accounting.
The first rollout-class source revision is preserved by source hash.

## Fifteen physical tables were not a complete device-model snapshot

`runs/pen_gpu_live_step_audit_v1/report.json` compares fresh actual task traces
with pre-start clones sharing the exact same readonly live device-model
object. Hashes of392 captured array fields remain unchanged during cloning.
The first20–40ms can be extremely close, but100ms still fails16/32 total
checks. Identical-input worlds within a batch also spread. Model reconstruction
alone therefore cannot explain the remaining discrepancy.

Comparison with native setConst/put_model finds seven differing derived fields:
actuator_acc0, body_invweight0, body_subtreemass, dof_invweight0, geom_rbound,
light_poscom0 and stat.meaninertia. Some absolute discrepancies (0.03125 in
inverse-weight fields) are small relative to their magnitudes. This comparison
is not a causal explanation of trajectory drift. The prior15-table artifact
is still valid as its declared physical subset; it must not be described as
a byte-exact GPU model reconstruction. Original reports are preserved.

`runtime/gpu_model_snapshot.py` now traverses all nested dataclasses/tuples
and captures404 device-array fields plus195 scalar/option/callback leaves.
An independent model requires exact scalar, key, shape and dtype compatibility
before ANY array mutation; unsupported leaves/callbacks fail. Restored arrays
are hash-checked exactly, including derived constants. Installed engine,
physics/material parameters, CUDA and drivers are not patched.

An initial snapshot-development attempt rejected a NumPy scalar before any
actual task execution. Its source/intent/log remain preserved. NumPy scalar
serialization is now explicit and tested. Successful v2 is retained; v3 adds
per-tick capacity, no-aid/mocap and exact initial warmstart/time receipts under
the same protocol. No favorable cohort is selected.

## Exact model restoration does not restore trajectory repeatability

`runs/pen_step_repeatability_v3/report.json` uses both dt, live-readonly and
full-array-snapshot bindings,128 identical worlds, three same-buffer resets
per binding and exactly the same100ms raw actions/reference. No stochastic
forces or parameter perturbations are injected. Fresh actual task cases are
run in the unchanged evaluator and still terminate at0.22/0.20s.

All156/156 integrity checks pass: exact model restoration/hash preservation,
initial pose/velocity/EMA/warmstart/time, original PD/clock, finite values,
contact/constraint capacity, zero external aid and fixed mocap. Dynamics
fidelity passes only24/72: hand q and velocity pass, while object position,
rotation and both velocity gates fail in every probe. Readiness remains false.

Worst100ms discrepancies across both bindings and three repeats:

| Quantity |5ms|2.5ms|
|---|---:|---:|
| Clone vs fresh actual object translation |3.281mm|1.648mm|
| Clone vs actual object rotation |0.17064rad|0.11937rad|
| Identical-world translation spread from world0 |3.440mm|1.648mm|
| Identical-world rotation spread from world0 |0.16645rad|0.11938rad|
| Same-buffer world0 repeat translation error |1.698mm|1.024mm|
| Same-buffer world0 repeat rotation error |0.08119rad|0.07496rad|

These are observed non-repeatability of THIS reset/step execution path.
They do not prove every simulator configuration is nondeterministic, identify
a unique faulty kernel, establish a calibrated physical uncertainty model,
or justify relaxed physics/trajectory gates. Hidden workspace behavior and
numerical contact/solver ordering still need causal isolation.

Installed primary source has atomic EFC row allocation in
`mujoco_warp/_src/constraint.py` and the upstream `forward_test.py` explicitly
sorts EFC rows for comparisons. These are plausible numerical-order mechanisms,
not a demonstrated sole cause. This phase changes no installed engine files.

`runs/pen_step_repeatability_review_v2/report.json` independently reconstructs
states, clock/control, initial conditions, capacity/no-aid/mocap, world-frame
angular velocities, SO(3) errors, spreads, repeats and the unchanged FAILED
gates:192/192 saved checks pass. Max readback discrepancy is5.96e-8rad (the
independent rotation metric uses rotation composition rather than dot/acos).
The runtime GPU hash check is not independently rerun by this CPU reviewer;
its source and snapshot pins are explicit.163 repository tests pass, including
nested snapshot restoration and rejecting a bad late field before ANY mutation.
The test suite is CPU-scoped; actual RTX execution evidence is in separate
physical run logs, not inferred from green unit tests.

## Reproduction and remaining work

From this repository, use the existing external Hand2 environment:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/audit_gpu_pen_steps.py --config configs/pen_gpu_step_audit_v1.toml --output runs/FRESH_STEP_AUDIT
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/audit_live_pen_clones.py --config configs/pen_gpu_live_step_audit_v1.toml --output runs/FRESH_LIVE_AUDIT
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/audit_pen_step_repeatability.py --config configs/pen_step_repeatability_v1.toml --output runs/FRESH_REPEATABILITY
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/review_pen_repeatability.py --report runs/FRESH_REPEATABILITY/report.json --output runs/FRESH_REVIEW
```

Frozen v1/v2 source aliases are recorded in provenance; current commands use
the maintained rollout class and current metadata. All raw previous failures,
models, traces and threshold configs remain unchanged. Large arrays stay in
ignored runs; manifests pin them. Historical archives/environments are intact.

Next isolate fresh-data-buffer versus same-buffer reset behavior and align
actual contact/constraint rows at the first divergence. Then test prediction
uncertainty under independent bounded action interventions before relying on
100ms planning or physical action-gradient supervision. An ensemble or shorter
horizon is a new hypothesis requiring its own validation; it cannot turn this
failed equivalence test into a pass. Continuous no-aid task execution remains
the final check. Do not fit these diagnostic failed prefixes as experts.

Original controlled learning/save-resume, matched residual/support objectives,
WM contact/action-gradient validation, independent ego RGB, multi-trajectory
total costs and hardware contracts remain. No neural checkpoint was created
this phase. Future neural training stops at its first healthy checkpoint for
restore verification, ETA and the user-requested Goal pause.
