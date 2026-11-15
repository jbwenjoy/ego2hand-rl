# Native CCD tolerance removes the recorded local presence split; dynamic benefit untested

Previous goal turn was PROGRESS: repeated source reconstruction reproduced
large20ms state/force differences, and20 collision-only queries reproduced
the final-step presence split at two near-identical input poses. This turn
executes pair-stage investigation v0.79 and native numerical sensitivity v0.80.
The full Goal remains ACTIVE/incomplete. There is no new neural/Actor update,
successful expert, stable pen task, real ego or acceleration result.

## Native coarse candidate is present in all queries

All tests keep the original saved model/geometry/physics unless explicitly
listed as a CCD numerical variant. Public arrays match exact saved layouts;
Data static capacities remain reconstructed, not original observations. No
installed source, driver/CUDA, old environment, archive or old result was edited.

The source pair is CYLINDER(5)/MESH(7): pen geom1 and
`robot/right_finger2_link4_col` geom29. It is not a discrete box/mesh pair.
The installed GJK function's discrete-geometry tolerance rule therefore cannot
be applied to this source. An initial BOX/MESH-only diagnostic guard rejected
the source after three successful box calibrations. That failed setup and its
source snapshot are retained in `provenance/pen_contact_stage_setup_v1`.
It performed four native collision queries (three fixtures plus first source)
and three direct phase queries, zero integration; no source packet was saved.

Corrected type coverage adds cylinder versus a convex box mesh at three known
center separations, in addition to three box/box fixtures. All six analytic
fixtures pass24/24 checks for native/phase contact presence, penetration depth
and overflow. Calibration is limited to these fixtures, not complex-mesh proof.

Staged pipeline uses the installed NXN broadphase and original narrowphase,
with its otherwise private pair arrays exposed. All20 recorded queries contain
physical pair1/29, order[1,29], explicit-pair/sensor IDs[-1,-1], broadphase row12
in this run. Total candidate count13 remains below192 capacity. Canonical
pair identity, not row number alone, establishes that it was not culled.
Native final presence and signed distances exactly match all previous outputs.
This rules out coarse candidate omission for these recorded inputs only.

## Separate phase wrapper remains unqualified

`runs/pen_contact_stages_v2/report.json` passes80/80 native staged-output/
restore/capacity checks, but its separate GJK/EPA wrapper's pre-manifold
contact flags match only10/20 native source outcomes. That limitation is
recorded separately;80/80 is not a global phase qualification.

Fresh Geom structures and explicit support-cache fields were then added
before the second GJK call. `runs/pen_contact_stages_v3/report.json` retains
130/160 mixed checks:10 contact-presence,10 contact-distance and10 raw-versus-
phase-distance checks fail. All cache fields before/after the raw call and
fresh phase initialization remain[-1,-1]. Fresh structures do not remove the
discrepancy; the cache-contamination explanation is not supported by this test.

For the contact-producing input, the separate raw call returns zero distance
with simplex dimension4, while the nested phase wrapper returns+34.176um and
does not enter EPA. For the absent-contact input they return about+1.795 and
+1.858um. Native contact remains about-17.118um for the former and absent for
the latter. These wrapper observations are NOT exact internal traces of the
native collision kernel. Compiler/inlining/arithmetic-context effects are
possible avenues, not an established cause; no iteration count or termination
reason is fabricated. The module is explicitly marked UNQUALIFIED and is not
used for task forces, action selection or the following numerical experiment.
All failed snapshots/reports remain unchanged rather than being overwritten
with a passing version.

## Predeclared native-only2x2 numerical experiment

Plan v0.80/config `configs/pen_native_ccd_factorial_v1.toml` and collector
321bc5c were committed before collection. Use only installed native
kinematics+collision, with no separate phase probe, force solve or integration.
CCD iteration cap50/100 and device tolerance1e-6/1e-7 form four cells.
Each cell queries all four recorded labels five times. They are TWO unique
qpos images, not four independent observations or confidence samples.

`runs/pen_native_ccd_factorial_v1/report.json`, SHA256
`7ac252564ab91d2e299b704498106cfab7a6bc8dbb84422b5dd6d4bdcc32700b`:

|CCD tolerance|iteration cap|original actual|instrumented R0|instrumented R1|instrumented R2|
|---|---:|---:|---:|---:|---:|
|1e-6|50|0/5|5/5|5/5|0/5|
|1e-6|100|0/5|5/5|5/5|0/5|
|1e-7|50|5/5|5/5|5/5|5/5|
|1e-7|100|5/5|5/5|5/5|5/5|

Entries count pair contact presence, NOT detector accuracy. Increasing the
iteration cap alone leaves this presence split unchanged. Tightening tolerance
changes presence for the absent-contact image at both iteration caps. All
fixed-pose repeats are byte-identical in recorded pair count/distance. The
two1e-7 cells give distances-17.76338um (original actual/R2) and-17.67001um
(R0/R1); cap100 yields the same results as50 here. CPU unfiltered geometric
comparators were about-17.196um for both images, with known geometry/precision/
filtering limits. Neither comparator nor increased contact count establishes
global accuracy or calibrated physical contact forces.

The baseline cell exactly reproduces the original20 outputs. Full model diffs
show ONLY `opt.ccd_tolerance` array and/or `opt.ccd_iterations` scalar change
as declared; all other404-array/195-scalar fields, geometry, material, mass,
inertia, actuators and constraint-solver policy stay identical. Actual device
tolerance values are9.999999975e-7 and1.000000012e-7 in float32. The device
override is explicit and saved; no installed library patch or hidden default
change occurs. Every cell's full resulting model tables/scalars are saved,
and remain unchanged after queries. Counts remain13 broadphase candidates and
0/1 total contacts in this diagnostic scene; no capacity/overflow failure.

Independent `runs/pen_native_ccd_factorial_review_v1/report.json` recomputes
model diffs, requested/observed values, raw contact arrays, baseline agreement
and per-pose repeated outputs:676/676 checks pass. Its native inputs are the
80 saved collision queries, not the unqualified GJK/EPA wrapper. No contact
force label is read from stale EFC buffers. Local numerical sensitivity is
established; stable trajectories, broader contact-domain fidelity and policy
benefit remain untested. The original failure is not rewritten as a success.

## Cost, validation and next action

This turn's native collision queries total136: failed setup4, corrected stage
v2/v3 each26 (six fixtures+20 source), native-only factorial80. The separate
phase wrapper runs55 queries (3+26+26). All runs add ZERO physics integrations,
force solves, optimizer or Actor steps. Initial compilation, including new
cap100 kernels, is not a training/GPU-utilization or speedup benchmark.

206 regression tests pass. Actual CUDA calibration/source queries supplement
those CPU-isolated unit checks. Compact native review, failed stage evidence,
source identities, test log and verified artifact ledger are retained under
`provenance/pen_contact_stages_results`; large packets/snapshots remain ignored.
All handles are terminal; no training/checkpoint is waiting.

Next test50/1e-7 as a DEVELOPMENT candidate against50/1e-6 on the full saved
seven-anchor57-action domain, repeated matched20ms execution paths and
independent EFC/momentum gates, then fresh original-clock actual control only
if justified. See v0.81. This candidate was selected from a tiny development
diagnostic, so the next cohort is not an independent claim of real-world
generalization. Original missing task imports, reconstructed static capacities
and all task/prediction failures stay explicit.

The full original learned SuperMimic/CHORD/save-resume, fair support-policy and
multi-trajectory costs, world-model contact/gradient validation, unified cube/
pen evaluation, true ego input/frame/asset/dynamics and hardware observation/
zero/gain/latency contracts remain required. No numerical presence probe
replaces them. Goal ACTIVE/incomplete.
