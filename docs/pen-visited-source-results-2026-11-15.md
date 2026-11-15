# Repeated public-state replay splits with a reproducible contact-generation change

Previous turn was PROGRESS: the fixed velocity cost failed to improve control,
and every failed forecast was diagnosed by physics tick. This turn executes
v0.78's saved-source investigation plus a predeclared collision-only probe.
The next action changes from extending the planning horizon to inspecting
contact generation. Full Goal remains ACTIVE/incomplete; no learned update,
successful expert, stable task, real ego or acceleration claim is made.

## Exact public arrays, explicitly reconstructed static metadata

Commit aa15dc7 and `configs/pen_visited_source_replay_v1.toml` precede collection.
Seven fixed anchors: velocity+support5ms at controls0/8/9/10, and state-only
2.5ms at0/1/2. The failed anchors are9 and1, with initial/adjacent passing
controls included, not selected afterward for favorable results.

Each independent one-world device model exactly matches all404 saved arrays
and195 scalar leaves. Each source import validates all169 Data array keys,
shapes and dtypes BEFORE any mutation, copies unused scratch NaN payloads
without repair, checks exact bytes and retains destination pointers. All live
reconstructed source arrays/models remain read-only around probes. Source
reconstruction/isolation35/35 and independent saved contracts294/294 pass.

Old controller packets omitted original Data scalar JSON. The new builder
derives nworld1, naconmax192 and njmax4096 from array dimensions; other values
come from the compatible installed constructor. All169 array layouts match,
but naccdmax/default-derived values are not independently observed in the old
packets. Fresh scalar/layout receipts are explicitly RECONSTRUCTED_NOT_ORIGINAL,
not retroactive observations. Original lost task-import enumeration also stays
unavailable. These limitations prevent a global original-state qualification.

Shared source-import code now rejects incompatible late fields before any
write and preserves exact scratch bytes. The existing response recorder gains
optional pre/post solver/warmstart instrumentation; its old bytes are archived
with an explicit mapping before editing. Fresh native-episode collection now
writes observed Data scalar/layout metadata for future runs. That new metadata
is not backfilled into old episodes or claimed exercised by reconstructed probes.
Shared implementation stays in src; installed packages, driver/CUDA, historical
environments and original run data are unchanged.

## Two matched execution paths, all repeats retained

`runs/pen_visited_source_replay_v1/report.json`, SHA256
`c1de77afdca098d1c68dfe069300ac551cea35a990ffb34cdc30f65665aa3e27`:
each anchor runs three repeats of the original57-candidate native graph, with
the SAME complete saved PD target table, EMA/reference/score inputs and original
candidate order. Every candidate's state/control/wrench prediction is retained.
Comparisons always inspect the ORIGINAL selected candidate, even if rescoring
chooses another action; changed choices are reported rather than executed.

Three further held-original-selected-PD repeats per anchor use an independent
single-step graph, with host readback/instrumentation between physics steps.
They save contact indices/EFC rows, moving-COM wrenches, pre/post solver counts,
warmstarts, no-aid/mocap/finite/capacity checks. This path has different
instrumentation/synchronization and is not assumed identical to the graph path.
Independent EFC force/torque/impulse review passes147/147 for all21 instrumented
rollouts. That is force-label consistency, not accurate predictions or control.

Total native work:21 candidate searches x57 rollouts =1197 candidate20ms
trajectories,6840 candidate physics ticks +120 instrumented ticks =6960 native
integrations. No actual episode, policy update, horizon extension or new expert
is created. All data are DIAGNOSTIC_NEVER_FIT.

Independent review `runs/pen_visited_source_review_v1/report.json` retains:

|comparison, original selected action|pass/total at all original gates|
|---|---:|
|candidate replay vs original actual|20/21|
|candidate replay vs original forecast|16/21|
|instrumented replay vs original actual|19/21|
|instrumented replay vs original forecast|17/21|
|candidate vs paired instrumented replay|18/21|
|candidate repeat1/2 vs repeat0|13/14|
|instrumented repeat1/2 vs repeat0|13/14|

Self-comparisons of repeat0 are excluded from repeat counts. Across ALL57
candidates, repeated forecasts pass770/798 comparisons;28 failures remain.
Even at the initial coarse source, some nonselected candidates fail while
the original selected action passes. A qualified baseline action does not
qualify its full local action domain or a model gradient. The review saves
all failures and exits1 by design; no gates were relaxed.

## The180ms source reproduces two branches

At5ms/control9, original selected index3 changes to2/2/3 across repeated
searches with the same restored public model/Data arrays and candidate table.
The selected-index3 trajectory matches original actual in two repeats and
original forecast in one. The instrumented path matches original forecast
in two repeats and actual in one. Paired paths differ in all three repeats;
thus instrumentation is not a universally stable replacement execution path.

Repeat-to-repeat selected discrepancies reach0.464814N force,0.0148497Nm
torque,0.038332rad object rotation,0.077469m/s linear and7.66653rad/s angular
velocity, with controls exactly equal. Nearby coarse sources160/200ms and
the initial source pass selected-action comparisons. At the fine failed source,
all new repeats match original actual but still fail the OLD forecast's force
gate. Replaying a prefix later does not repair the original failed prediction.

In instrumented source180ms repeats0/1, the fourth physics tick creates a
pen/index-finger-link4 contact (geom1/29), with about17.118 micrometers signed
penetration and0.38079N local normal force. Repeat2 and original actual have
no object contact there. Solver counts are4/4/1/3 versus4/4/1/1. Before that
last step, recorded inputs differ only in one hand joint by2.98023e-8rad and
object z by4.65661e-10m. Contact/solver pattern association alone does not prove
a solver bug or complete hidden-workspace reset. Installed solver/collision
contexts allocate private arrays outside public Data; their existence is not
proof that stale scratch caused these differences. Read-only primary source
bytes and inspection notes are preserved separately.

## Isolated collision generation reproduces the local split

Commit2218229 and `configs/pen_contact_branch_v1.toml` predeclare a zero-
integration probe of all four recorded last-step inputs: original actual and
instrumented repeats0/1/2. They contain TWO distinct qpos images, not four
independent observations. Each input is applied to restored public arrays,
then five repeated graphs run ONLY kinematics and collision detection. No
constraint solve or integration runs, and no forces are read from stale EFC.

`runs/pen_contact_branch_v1/report.json`, SHA256
`1658080f7def682e11c42ab84fe648c8b4f72dce3196e3febac8d3df180d1eba`:
all20 queries preserve exact source-array/scalar resets, the404/195 model,
and capacity (13 broadphase candidates versus192 capacity). Fixed-input pair
presence is identical in all five repeats. The two slightly different poses
give the following stable outputs:

|recorded input|GPU pair1/29 presence|CPU direct signed distance|
|---|---:|---:|
|original actual|0/5|-17.1959501 micrometers|
|instrumented R0|5/5|-17.1951450 micrometers|
|instrumented R1|5/5|-17.1951450 micrometers|
|instrumented R2|0/5|-17.1959501 micrometers|

CPU query uses detached topology with explicit captured GPU geometry-table
values, actual mocap and the recorded qpos. It is an UNFILTERED direct distance,
not a filtered-contact or force truth oracle. CPU/GPU geom-position difference
is at most3.011e-8m; differing precision, geometry implementation and filtering
still need separate investigation. Negative CPU distance in all four inputs
does not alone prove a universal GPU false-negative bug or validate CPU dynamics.

This local contact-presence difference can be reproduced BEFORE constraint
solving, so solver/workspace hypotheses alone do not explain the two recorded
kinematic inputs. Exact fixed-pose collision queries are stable in this sample;
the observed whole-step repetition split involves small preceding state changes
and sensitive contact generation. No broad deterministic-backend conclusion,
global error bound or physical robustness fix is established.

The inspected figure is `runs/pen_visited_source_figure_v2/summary.png`, with
pinned inputs/source. It shows original/repeated forces and the collision-only
table; the first figure's renderer bytes are retained before a header-layout
adjustment. No dynamics or scientific data changed during plotting.

## Next action and full objective

206 tests pass, including late-layout rejection without partial writes and
NaN-payload/pointer preservation. Actual RTX runs separately establish the
native evidence; CPU-isolated tests do not substitute for it. Compact reviews,
figure metadata, installed-source inspection and explicit pin mappings are in
`provenance/pen_visited_source_results`. Large arrays/figures stay ignored.
All current processes are terminal; no training/checkpoint wait exists.

Next inspect pair-level broadphase and convex-distance/contact generation on
these fixed recorded inputs before any horizon expansion or new neural epochs.
See v0.79. Original task failures, lost import list and reconstructed static-
metadata limits remain. The full Goal still requires learned SuperMimic/CHORD
control/save-resume, fair support-policy comparisons and gradient/contact
validation, unified cube/pen evaluation, shared multi-trajectory total cost,
true ego input/frame/assets/dynamics evidence and hardware observations/
zero/gain/latency before deployment. Current diagnostics replace none of those.
