# Original cube and single-pen unified no-reset results

**Later correction:** [GPU contact-label audit](contact-label-correction-2026-11-14.md)
found copied host `adhesion` residue contaminating normal forces in this v3 run.
Its force readback6/6 reproduces those labels, not GPU force truth. Original
kinematic/geometry failures stand; corrected v4 data and independent27/28 EFC
review are separate. Do not fit affected v3 force labels as trusted targets.

2026-11-14. Previous goal turn was progress (ca54864). This turn implements and
runs the v0.58 controlled milestone through the unified source on RTX5090.
Four fixed cases complete normally in the final receipt run. None qualifies;
the full Goal remains active and no neural update was performed.

## Executed comparison

`runs/controlled_evaluation_v3/report.json` is authoritative. Cube uses the
original iteration16 smoke checkpoint, one seeded fixed SO(3) goal and2s.
Pen uses iteration12, exactly `01_single_axis/clip_001`,369 frames/7.36s, first
frame hold disabled. ZERO is the original task's zero NORMALIZED action, not
zero torque, zero PD target or a mug-style residual about a reference grasp.
Its mapping/EMA are unchanged task implementations. These are smoke weights,
not successful experts. This is a diagnostic, not a matched A1/A2 experiment.

| Task/controller | Executed duration | Planned duration | COM RMS | Full rotation RMS | Maximum native penetration |
|---|---:|---:|---:|---:|---:|
| Cube ZERO | 2.000s | 2.000s | 23.534mm | 2.4361rad | 4.5701mm |
| Cube saved Actor | 2.000s | 2.000s | 24.481mm | 2.4596rad | 4.5701mm |
| Pen ZERO | 0.160s | 7.360s | 29.844mm | 1.5692rad | 3.8012mm |
| Pen saved Actor | 0.160s | 7.360s | 29.602mm | 1.5811rad | 3.8012mm |

Cube runs reach2s without native termination but fail declared position,
orientation and penetration gates. Pen runs stop on the first native50mm
position-error termination; shaft orientation termination is false, but full
orientation error is large. The latter distinction matters: the original pen
shaft-only reward/termination does not establish full6D tracking. Neither
controller is better in a way that solves the tasks or supports a method claim.

No reset occurs after the single initialization. There are200/200/32/32 physics
steps (cube10ms,pen5ms), with original50Hz actions and original solver/actuator
semantics. No step/interval perturbation events remain; generalized/body applied
forces are zero at every recorded physics step. Fixed wrist mocap never moves.
Complete initial integration state, source/endpoint pose and velocity, actual
controls/actuator forces, GPU clock, observations, actions, all native termination
flags and contact/constraint capacities are saved. Pen reference phase1 is the
next20ms goal after frame0 reset; elapsed state tracking uses the original
369-frame clock without extrapolation or holding at clip end.

Both task pairs have byte-identical complete initial integration states and
full reference arrays. All15 saved GPU physical parameter tables match within
each pair; their compiled host models also hash identically. GPU physical tables
are separate from the host topology binary: that binary alone is not an exact
GPU physics replay/resume checkpoint. Native warning counters are unavailable
through the Warp copy; the report uses explicit finite/capacity checks and
does not report native warning_count=0 or the original mug's static tripod gate.

Final case-loop wall time12.261356s includes case environment construction,
step recording and comparisons but excludes preceding imports/CUDA setup and
earlier diagnostics. Full per-physics-step host copies are deliberate observation
costs; no performance/real-time or shared-model amortization conclusion follows.

## Pairing correction and raw review

v1 retains four actual failed task cases, but the pen initial qpos differs by
up to1.19e-7 and qvel1.26e-5 between separately constructed environments.
Its exact pairing gate fails, so it is not the authoritative paired comparison.
v2 uses an explicit whole-run `torch.jit.optimized_execution(False)` context;
initial qpos/qvel/reference pairs then match exactly. This is evidence consistent
with different script-transform execution paths; the exact cause was not isolated.
No numerical tolerance was relaxed. v3 retains v2 settings and adds the complete
receipts listed above, strengthening pairing to include the full integration state.
v1/v2 source/report/log snapshots and all original run directories are preserved.

`runs/controlled_evaluation_review_v1/report.json`:6/6 independent readback
checks pass. Raw contact normal forces/penetrations and observed COM reproduce
exactly; metric disagreement is at most2.22e-15. CPU Actor inference from saved
pre-action observations agrees with recorded GPU actions within4.47e-8. The
review verifies no trajectory jump, no continuation after earlier flags, legal
controls, phase/duration clocks and unchanged failed qualifications. This is
validation of recorded failures, not six physical task successes.

The reviewer also reconstructs full contact force+couple wrenches about the
moving PRE COM in world and object BODY axes, saving both per-tick and20ms means.
These are diagnostic actual-contact labels, not successful support targets or
WM/action-gradient validation. The simple free-object linear impulse closure
has maximum residual about4.24e-4N*s for cube versus below4e-9N*s for pen.
Cube's larger discrepancy remains unexplained; resolve force/solver/integrator
accounting before using those labels to certify a learned contact model.

## Initial contact geometry and next action

`runs/controlled_initial_geometry_v1/report.json` audits the saved initial state
using fresh native forward-only geometry with the captured GPU physical tables.
It preserves qpos/qvel/mocap and performs no integration. Cube has no initial
contacts. Pen has a deepest overlap of2.881998mm at finger3's tip sensor against
the pen shaft, exceeding the declared2mm execution threshold before learning.
During actual GPU execution, the pen's first5ms step reaches3.801205mm at
finger2's tip; cube reaches4.570119mm at70ms against thumb link3. The forward
audit is separate native geometry evidence, not reconstructed GPU solver force.

The next prerequisite is a physically qualified controlled initialization and
an audited action/reference-to-PD contract, while retaining raw reference poses,
velocities and clocks. Check how the original action zero and EMA relate to the
reference grasp before attributing early release solely to learning. Any bounded
contact-aware initialization correction must be declared separately and assessed
without silently replacing these failed baselines. Resolve cube impulse accounting
before fitting its contact labels. Do not simply increase iterations on these
failed states or label their packets as successful expert demonstrations.

## Assets, playback and full remaining scope

Optional explicit cube XML and jig mesh arguments remove source-adjacent asset
dependence without copying assets into Git or changing historical source/envs.
The cube import-time missing-default assertion is deferred to actual loading so
an explicit asset can be supplied. Original defaults/geometry/PD physics remain.
`runs/controlled_cube_asset_equivalence_v1/report.json` passes4/4 comparisons
against immutable archive code: baseline/45mm/65mm cube and full hand+rig model.
Actual external XML/texture/jig bytes are pinned via the artifact registry.

`runs/controlled_evaluation_video_v1` contains both ZERO/Actor actual replays,
with saved wrist mocap, honest physical clocks and terminal stills. Cube video:
289 frames at4x slow motion; pen:97 frames at10x slow motion; both1280x800/30fps.
First/middle/last decoded frames were checked. No physical integration occurs
during rendering. Full148 tests pass, including rejection of perfect short
prefixes, hidden force/overflow and shaft-preserving full pen roll error.

Original successful controlled learning/no-aid evaluation, object-frame support,
matched A1/A2 and action gradients, ego asset/dynamics/frame/controller contracts,
independent RGB/full/multiple trajectories/total costs and hardware contracts
remain incomplete. Existing training/restore smoke evidence is preserved, not
promoted to task success. No new training started, so the requested first-healthy-
checkpoint training pause boundary is not triggered. No driver/CUDA changes.
