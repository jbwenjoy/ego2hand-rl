# Bounded GPU dynamic-wrench proposals and actual execution

Previous goal turn made progress by exporting verified GPU playback. This
phase archives that milestone in fc69031 and advances from geometry-only
correction to actual dynamic-wrench-aware contact proposals. A new initial
pose passes first-frame support checks, but neither physical trajectory
passes the task. The full research Goal remains active. No neural training,
driver/CUDA change or modification of historical study trees occurred.

## Resident batched physical queries

`runtime/gpu_contact_oracle.py` reconstructs the saved model with all15 actual
GPU physical tables. Named joint/actuator indexing preserves the object-first
layout. It validates free-object/zero-COM, all-joint actuation and no-aid
contracts. Fixed state/model tensors stay on GPU. The repeated forward work
uses a CUDA graph on a shared Torch/Warp stream; each query preserves original
object/wrist and source velocity, with explicitly zero solver warmstart.
There is no integration in these queries. One packed output/state readback
per batch avoids per-world/contact host synchronizations in the query loop.

`runs/pen_gpu_oracle_audit_v1/report.json`:9/11 checks pass. The two failures
are the declared finite-difference directional checks, not hidden or relaxed.
The actual active-contact set changes under h/2 perturbation; full-axis J*v,
directional h and h/2 derivatives disagree substantially. No stable local
simulator derivative or learned world-model action gradient is established.
The next proposal search therefore uses no finite-difference Jacobian.

Saved static wrench replay errors are4.00e-8N and7.26e-10Nm. Independent actual
EFC-to-COM-wrench identities pass in static and moving queries. Center and
plus/minus1e-3rad perturbations of20 q and20 ctrl give81 query points. Across
three identical sweeps, batch versus individual maximum discrepancies are
9.01e-5N and1.94e-6Nm, within the preregistered1e-4N/1e-5Nm tolerances but
not byte-exact. Original velocities/nonjoint state, legal bounds, capacity,
finiteness and zero elapsed physics time pass.

| Same81-query sweep | Wall times, seconds | Median |
|---|---|---:|
| Batch81 | .001316 / .001358 / .001224 | .001316 |
| Individual81 | .080597 / .081194 / .080474 | .080597 |

The observed ratio is61.25x for this narrow query workload. Both include state
writes, captured forward, output conversion and CPU readback; setup, compile,
contact audit and file IO are excluded. Setup/warmup is2.155s. This comparison
does not measure training utilization, learning updates, full retargeting cost
or generalized acceleration. The report records every explicit query count.

## First-frame actual force AND moment proposal

`runs/pen_gpu_support_search_v1/report.json` follows v0.63/config exactly:
seed6201,20 populations of512,32 elites, no restarts. Variables are20 joint
offsets within0.1rad of source and20 independent PD preload offsets within
0.05rad of the proposed pose, additionally bounded by original hard/ctrl
limits and0.1rad residual about qref0. All object/wrist, gravity, material,
inertia, source velocity and original-clock target inputs remain fixed.

The score uses actual BODY-COM wrench error, full6D nominal dynamic demand,
source joint acceleration mismatch, small joint regularization, penetration
and nonhand-contact penalties. The declared scales are0.03N/0.0015Nm and
2mm object/0.2mm self penetration. These select proposals; they do not loosen
the original full-task tracking gates. Every candidate's bounded inputs,
wrench, acceleration, contact metrics and score is saved, including failures.

10240 search slots,512 explicit repeated-center preflight slots and2 selected
single-world queries were used. Each oracle also has two setup forward API
calls (warmup and graph capture), outside candidate counts. Setup/preflight
takes1.621s; the search loop including saved populations takes3.183s. Full
CLI time and this component time are not a matched end-to-end speed comparison.

The selected pose changes joints by at most0.08155rad; maximum physical
residual is0.09590rad and preload0.04237rad. Actual moving contact force error
is0.00923676N; moment error is0.000088035Nm. All63 object-hand and189 self
native convex-distance queries pass the unchanged geometry limits; maximum
object overlap is1.97580mm, very near the2mm boundary, self overlap zero.
This remains ideal simulator geometry rather than calibrated real clearance.

The pose uses fourth-finger contact as well as thumb/third finger. There are
five allocated/constraint-active object contacts, but two have zero normal
force; do not call them five load-bearing contacts. The original geometry-only
diagnostic used only first-three-finger tip/link4 eligible support. This search
explicitly permits all existing robot collision geoms. No contact geometry,
material or actuator authority is added. The shared initializer has a new
explicit all-robot-support option; its default behavior is preserved.

The moving full6D cone, static full6D cone and bounded-PD static LP now pass.
LP ray coefficients are numerical witnesses, never applied external forces.
`runs/pen_gpu_support_review_v1/report.json`:17/17 independent saved checks
pass, including reconstruction of all10240 scores/global selection, EFC
decoding, full252-pair geometry and moving/static LP primals. Static smooth
forces/Jacobians are independently evaluated in native double precision with
saved actual GPU contacts and generalized forces. No physics is integrated.

These targets remain conditional on nominal inertia and four-point source
derivatives, with the previously documented initial-velocity inconsistency.
No measured source force or full-reference feasibility follows.

## Actual original-clock GPU execution still fails

`runs/controlled_support_pen_GPU_v1/report.json` applies the selected joint
pose and independent constant PD residual, initializing original EMA/raw
action accordingly. It uses the unchanged50Hz reference phase1 after reset,
source velocities, first task termination, no post-start resets or object aid.
Both dt start from the same selected state/command. All actual robot contacts
are explicitly included in the support diagnostic. No Actor is constructed.

| GPU dt | Old original / geometry duration | New duration | New COM RMS | Max overlap |
|---|---:|---:|---:|---:|
|5ms| .16 / .16s | .22s |27.316mm|1.97583mm|
|2.5ms| .16 / .16s | .20s |24.291mm|1.97668mm|

Both new trajectories pass penetration/self/joint/no-aid/capacity gates but
fail position, orientation and complete7.36s duration. This is only a short
diagnostic prefix improvement, not robust task success, learned Actor benefit,
or a matched training-budget A1/A2 comparison. Old and new source-pinned runs
are shown separately; no claim of exact deterministic GPU replay is made.

`runs/controlled_support_pen_GPU_review_v1/report.json` passes40/40 execution
checks, EFC14/14. It reconstructs original scale/EMA/PD targets (zero error),
source clock/velocity, carries, initial state, mocap/warmstart, full cone
primals and torque bounds, metrics (zero error) and unchanged failed gates.
161 repository tests pass, including missing-moment and invalid-capacity
proposal rejection. Those checks verify failures, not successful tasks.

## Where the first-frame improvement is lost

`runs/support_pen_wrench_drift_v1/report.json` compares saved actual PRE-step
EFC wrenches with the original nominal WORLD force/moment demand, interpolated
on the original clock and rotated into the ACTUAL object's BODY axes. It
does not compare vectors expressed in different moving frames.

The first execution wrench errors are0.01291N/0.0001258Nm. The original
phase1 first PD target differs from the search's qref0 target by up to
0.0059824rad. Both runs leave the proposal wrench tolerances by t=.005s;
hand normal force first drops below0.1N at .095/.0875s. These observations
locate the rapid loss of nominal force/moment tracking. They do not isolate
a single cause or validate the uncertain source derivatives. First-frame
force matching and constant residuals cannot qualify continuous contact.

Three-way actual playback is in `runs/controlled_support_pen_video_v1`:
`GPU_pen_three_way_0.005s.mp4` (181 frames) and
`GPU_pen_three_way_0.0025s.mp4` (169 frames),1536x800,30fps,20x slow motion.
Original, geometry-only and support-aware columns use saved physical tables,
wrist state and qpos, without interpolation/integration. Explicit terminal
stills show different durations; decoded first/middle/end images were checked.

## Reproduction and next phase

Using the existing external Hand2 Python from this repository, run:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/audit_gpu_contact_oracle.py --config configs/pen_gpu_oracle_v1.toml --output runs/FRESH_ORACLE
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/search_gpu_pen_support.py --config configs/pen_gpu_support_search_v1.toml --output runs/FRESH_SEARCH
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/review_gpu_pen_support.py --report runs/FRESH_SEARCH/report.json --output runs/FRESH_REVIEW
```

Configs explicitly pin local input reports. To use fresh upstream artifacts,
write a new config with those paths; do not silently rewrite frozen reports.
`tools/run_gpu_support_pen.py`, `review_gpu_support_execution.py`,
`analyze_support_wrench_drift.py` and `render_support_pen_comparison.py`
provide actual execution, saved audits, diagnostic clocks and playback.
Provenance retains report/test/video pins and the prior initializer's exact
source bytes, so old run pins resolve through the explicit source-alias ledger.

Next validate resident batched physical STEP/EMA/phase reproduction before
short-horizon moving-contact optimization. Use original phase1 on the first
action and preserve source velocity/warmstart/full state during planning
clones. Physical execution must remain a single no-reset trajectory. Do not
fit these failed diagnostic prefixes as experts. Successful controlled
learning/save-resume, matched residual/support objectives, WM contact/action
gradient validation, shared multi-trajectory total costs, independent ego RGB
validation and hardware contracts remain. Future neural training must stop at
the first healthy saved checkpoint, verify restore/ETA and pause as requested.
