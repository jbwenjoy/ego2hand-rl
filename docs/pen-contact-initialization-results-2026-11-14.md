# Pen geometry correction, native dynamics and BODY-frame support targets

Previous goal turn progressed in d93dde1, correcting copied-host GPU contact
labels. This turn progresses with one bounded projection, actual native
two-dt rollouts, indexed torsional-support LPs and original-clock dynamic
wrench targets. None is a new learned policy or successful full task. The
full original Goal remains active; no neural training started.

## Geometry correction is small and native-geometrically qualified

`runs/controlled_pen_projection_v1/report.json` implements preregistered v0.60.
Object pose and wrist are unchanged;20 named joint offsets are limited to
0.10rad and legal hard/PD ranges. Fingers1/2/3 tip/link4 proximity is declared
before optimization. One least-squares solve, no candidate/seed search.

All63 eligible object-hand and189 hand-self pairs are inspected. Initial
all-pair object penetration is3.801490mm, including index-finger overlap that
was absent from the old native forward contact list. The previous2.881998mm
forward-only report remains valid only for that listed contact set, not as an
all-pair clearance bound. Native convex distance queries and native forward
can return different contact activation; retain that discrepancy explicitly.

The proposal's maximum joint change is0.03408943rad. Maximum final all-pair
overlap is0.05004696mm; three intended group gaps are approximately-0.05mm.
All six final geometry checks pass, including re-querying every eligible pair
without sphere pruning, bounds and fixed nonjoint qpos. Solver40nfev/700
residual calls,105854 engine native-distance queries plus252 final unpruned
queries, and one final native forward readback. Solver wall0.6128s excludes
model setup and final qualification; this is not a full-cost speed claim. No integration. This
is compiled ideal-convex native geometry evidence, not independently calibrated
real CAD clearance or support feasibility.

## Actual moving-reference native execution still fails

`runs/controlled_projected_pen_native_v1/report.json` compares original and
projected initialization at5ms and2.5ms native CPU integration, original50Hz
369-frame/7.36s reference, original initial robot/object velocities. Projection
also declares a constant source-joint residual delta and matching initial EMA;
it is not an initial-state-only causal intervention. Original uses zero residual.
Raw action2*delta reproduces the original0.5 physical scale/0.5 EMA. No target
clipping occurs, no wrist movement, force aid, pose reinjection or time stretch.

| Variant | Native dt | Executed duration | Max native penetration | Qualified |
|---|---:|---:|---:|---|
| Original zero residual | 5ms | 0.16s | 6.1933mm | false |
| Original zero residual | 2.5ms | 0.16s | 5.1340mm | false |
| Projected constant residual | 5ms | 0.22s | 7.2471mm | false |
| Projected constant residual | 2.5ms | 0.18s | 6.0769mm | false |

All stop on POST endpoint position error exceeding50mm; full tracking,
penetration and complete-duration gates fail. They are not new RTX/Warp Actor
results: native CPU double precision uses captured GPU physical parameters.
The source GPU manager's termination sees one substep-stale derived state,
whereas this diagnostic checks actual POST endpoints, a declared difference.
CPU model allocation has no Warp contact-capacity gate; native warning counters
are explicitly recorded. These runs do not establish timestep robustness.

`runs/controlled_projected_pen_review_v1/report.json`:56/56 saved-data checks
pass. They reconstruct source velocities/clock/initialization, original EMA/PD
targets, chronology, first terminal, raw contacts, metrics and unchanged failed
gates. Max generalized-force error2.22e-16N and max impulse residual below
1.74e-18N*s confirm native copied-parameter force accounting on these prefixes.
These are checks of recorded failures, not56 manipulation successes.

Actual primary-dt replay is saved as
`runs/controlled_projected_pen_video_v1/native_pen_original_vs_projection.mp4`:
115 decoded1280x800/30fps frames,10x slow motion, explicit terminal stills.
It discloses the weaker fine-dt result and does not display model predictions.

## Full force AND moment support matters

`runs/controlled_pen_preload_v1/report.json` freezes projected geometry, sets
static velocities to zero for this separate necessary-support test, uses actual
native active contacts, condim4 torsional friction and original20 direct PD/
joint torque limits. Commands may differ at most0.05rad from projected joints.
Target is gravity support with zero moment at object COM, represented in BODY
axes. A feasible LP would only be a witness; its coefficients are never applied
as external forces. Here that full6D LP is infeasible, so no preload or claimed
compliant equilibrium is produced and no preload dynamics arm is launched.

Native forward activates only thumb/middle tip contacts, despite the native
distance proximity check also finding the index group near the shaft. The
12-ray wrench matrix has rank6; rank alone does not imply positive-cone support.
`runs/controlled_pen_support_diagnosis_v1/report.json` repeats the bounded LP,
then removes actuator bounds: full6D support remains infeasible. Removing the
moment equations yields force-only success with remaining COM moment
[-0.0005109,0.0012971,-0.0028979]Nm. That weaker result is NOT qualification.

`runs/controlled_pen_support_certificate_v1/report.json` independently checks
a separating witness: minimum ray dot-8.88e-16, target dot-1 (tolerance1e-9).
It certifies only this supplied frozen native cone/nominal target, not global
grasp/task infeasibility, a physical action gradient or a measured source force.

## Original moving reference gets its own dynamic target

Static gravity support must not replace the original dynamic demand. New
`mechanics/reference_contact_wrench.py` implements F=m(a-g), T=I_world*alpha+
omega cross(I_world*omega), rotating both to the object BODY axes about COM.
Principal inertia orientation is included. The nominal simulator mass is
0.0299999993kg, BODY diagonal inertia approximately[1.17044e-6,1.94579e-4,
1.94579e-4]kg*m^2. These are model priors, not measured ego-object parameters.

The declared4-sample forward position/SO3-log stencil estimates derivatives at
each ORIGINAL frame, rejects ambiguous pi log windows and uses no smoothing or
extrapolation.366 of369 frames have targets; last3 are explicitly invalid.
Analytic translation acceleration, gyroscopic couple and world-coordinate
rotation invariance tests pass before target generation.

`runs/controlled_pen_reference_wrench_v1/report.json`/`targets.npz` preserve
these targets and masks. First dynamic BODY wrench is approximately
[0.0148434,-0.0416086,0.2720344]N and[-8.33e-6,-0.00826169,0.00067782]Nm,
distinct from static gravity. Its frozen first-frame cone test also fails;
separating minimum ray dot-6.66e-16/target dot-1 is checked. This is conditional
on the nominal inertia and derivative estimator. The higher-order initial
velocity estimate differs from the task initializer's original forward
difference by about0.0192m/s in norm, so derivative uncertainty/alignment must
be addressed before treating nominal inverse-dynamics targets as reliable
supervision. No actuator-compliant moving-hand solution or full-reference
trajectory feasibility follows from this instantaneous test.

## Next phase and full remaining scope

Next compare actual GPU contact activation at this SAME projected pose with
the native two-contact picture, using the corrected GPU force reader. Do not
change contact bounds or infer global impossibility from native frozen cones.
Then address full dynamic wrench feasibility/contact arrangement, reference
derivative consistency and soft-contact/controller dynamics together; merely
clearing initial penetration does not solve retargeting. Preserve all failed
prefixes as diagnostics, not successful expert labels.

158 repository tests pass, including object-first joint indexing, insufficient
projection room remaining failed, torsional ray couples, full6D support with
actual PD authority, analytic inverse dynamics and coordinate invariance.
The existing static-feasibility and separating-witness core is reused rather
than copying its optimization math. All artifacts remain under the unified repo;
large states/assets remain external/ignored, historical archives/envs untouched.

Original successful controlled learning/save-resume/no-aid eval, object-frame
support objectives with matched A1/A2 and WM/action gradients, ego geometry/
inertia/material/gravity/frame/controller inputs, independent RGB/full/multiple
trajectories/total amortized costs and hardware subversion/observation contracts
remain required. No system driver/CUDA change or neural optimizer step occurred;
first-healthy-neural-checkpoint restore/ETA pause remains for future training.
