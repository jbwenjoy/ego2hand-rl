# Bounded pen contact initialization, separate geometry and dynamics

Previous turn progressed in d93dde1: corrected GPU force labels and exact pen
control audit,152 tests; four tasks still failed. Full Goal remains unchanged.
No neural training is started in this phase.

The saved initial pen has only finger3 in its native forward contact list, but
an all-eligible-pair native distance probe also finds finger2 penetration about
3.8015mm. This shows why contact-list-only reset inspection is insufficient.
Preserve the old forward-only2.882mm report with its limited scope; new geometry
proposal checks all63 object-hand and189 eligible hand-self pairs.

Preregister one bounded joint-only proposal on fixed clip_001/frame0. Fixed
object pose, wrist and source reference/clock. At most0.10rad per each20 named
joint, intersect actual hard/control limits. Contact intent is fingers1/2/3
link4 or tip-sensor collider, not chosen afterward by observed task outcomes.
Target50micron penetration at those groups, at most0.2mm qualification overlap
and0.2mm contact gap; other-hand/self clearances use0.2mm during proposal.
Native conservative sphere bounds only prune separated pairs; every eligible
pair is re-queried without pruning and native forward contacts inspected after
optimization. This is native convex geometry evidence, not independent CAD.
One SciPy bounded least-squares solve,1600 residual calls/45s/80nfev maximum;
retain the best completed proposal and all failures, no seed/candidate search.

Then compare original and proposed initial states under native CPU dynamics
at original5ms and half2.5ms, original50Hz clock and full7.36s/first native-style
50mm position or60degree shaft/numerical stop. This is a controlled diagnostic
copy using captured GPU physical tables, not a new GPU policy evaluation.
Preserve actual original object/robot velocities. Joint-only correction delta
is also the declared constant residual offset on the source joint reference,
with the original physical0.5 action scale/0.5 EMA and raw action2*delta;
initialize residual EMA to delta to match this new initial state. Original
baseline uses zero residual/zero memory. Preserve original absolute target
clamping and report any clipping. No object force, equality, gravity assistance,
pose injection after reset or source time stretch. Record all contacts and
control/EMA/source clock; geometry qualification cannot substitute for dynamics.

Use the existing shared tracking summary with unchanged5mm/10mm COM,5/10degree
angle,2mm penetration/0.2mm self/1mrad joints/0.1N nonhand gates. It is not the
mug static tripod gate. If correction fails, keep the failed pair rather than
increasing its bounds/loosening gates or training it as an expert. Native CPU
and Warp results must remain labeled separately.

Original controlled learning and save/resume/no-aid qualification, object-frame
support/matchedA1A2 and WM/action gradients, ego asset/dynamics/frame/controller
contracts, independent RGB/full/multiple trajectories/total costs, and hardware
subversion/observations remain required. First healthy neural checkpoint
restore/ETA pause remains in force when training is eventually launched.

## Fixed-geometry support follow-up before its execution

The single geometry solve passes at max0.03409rad joint change, but original
native dynamics stops0.16s at both dt; projected constant-residual execution
stops0.22/0.18s, with dynamic penetration still failing. Do not tune the geometry
bounds or label it a solution. Separate contact geometry from actuator preload:
one full6D object-BODY support LP on the FIXED projected pose, zero static
velocity, actual native active contacts/friction including torsional condim4.
Only first3 fingertip/link4 contacts may support it. Target is gravity support
with zero COM moment. Use all20 actual direct PD/joint force bounds and target
commands within0.05rad of projected joints; minimize total ray coefficients.

No LP coefficients are applied as forces. If feasible, invert the PD witness,
read back an actual native soft-contact solve at that command, and preserve any
acceleration/net-wrench mismatch. A later two-dt source-follow diagnostic may
reuse the projected initial pose with that CONSTANT controller residual and
matching EMA, restoring original source velocities. This changes control
preload as well as initial pose, and is not an initial-state-only causal claim.
No further LP search/candidate or threshold relaxation is authorized by a pass.

## Moving-reference wrench check, declared before its result

The fixed projected cone cannot balance a static full6D gravity wrench even
with actuator bounds removed; force-only support leaves a nonzero COM moment.
An independently checked separating witness applies only to that frozen cone.
Do not substitute this static wrench for the original moving-reference demand.

Build original-clock nominal inverse-dynamics targets with m(a-g) and
I_world*alpha + omega cross(I_world*omega), then rotate force/couple to object
BODY at COM. Use simulator mass/principal inertia/body-inertia orientation;
these are nominal priors, not measured source forces. A declared4-sample
forward position/SO3-log stencil estimates derivatives AT each original frame;
last3 frames are invalid, no smoothing, extrapolation or time stretch. Log
windows near the pi branch must be rejected. Preserve known reference motion
as input, not actual future executed states. Check analytic acceleration,
gyroscopic couple and world-coordinate rotation invariance before using targets.

Test only the first moving-reference target against the SAME saved frozen cone,
without changing geometry or acceptance. A pass is necessary wrench feasibility
only and does not include moving-hand torque limits, compliance, later contact
evolution or uncertainty from finite reference differences. A failure must retain
its numerical separating witness. This adds the dynamic support contract; it
does not change any preceding failed physical rollout or qualify an expert.
