# Controlled GPU contact-label correction and action audit

Previous goal turn was progress (3183a28): the original cube/single-pen no-reset
diagnostic and videos were saved, all four task cases failed. This turn checks
the controller contract and repairs an actual observation-label defect, without
new neural training. Full Goal remains active; successful controlled learning,
support/matchedA1A2/WM gradients and ego/generalization scope are not complete.

## Pen control contract is correct

`runs/controlled_pen_action_audit_v1/report.json` reconstructs both v3 pen cases
directly from source clip joint positions, saved raw actions and actual PD
commands. Both2/2 pass with maximum target difference exactly0. Reset joint
positions equal source frame0 exactly. ZERO means zero residual on the moving
reference, not zero joint angle or torque. Original scale0.5/EMA0.5 is preserved;
no absolute target clipping occurs in either executed prefix. Maximum first
reference-target change is0.005982399rad; Actor residual reaches0.007605794rad.
The v4 audit repeats2/2 exactly. This rules out the proposed wrong-zero-target
explanation for these prefixes; it does not establish feasible reference contact.

The original native forward geometry finding remains:2.881998mm pen/third-tip
overlap at reset, followed by3.801205mm actual first-step penetration. v3/v4
initial full integration states and GPU parameter tables match byte-for-byte,
so the initial geometry evidence applies to both. The next initialization work
must preserve original references/clock and declare any correction explicitly.

## Actual defect: copied host adhesion, not GPU momentum integration

Installed Warp `get_data_into` copies GPU contact geometry/constraint fields
into host MjData but does not fill the native contact `adhesion` field. Native
`mj_contactForce` subtracts that field from the decoded normal force. In this
actual run the copied field contains residual values, producing erroneous
positive or negative normal offsets despite all20 actuators being joint PD
transmissions and no adhesive actuation. The native net-normal subtraction is
documented in [MuJoCo3.11 engine_core_util.c](https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_core_util.c)
lines1001–1020; local dependency files and reproduction receipts are pinned.

The bounded original GPU cube12-tick audit separates actual dv/qacc, generalized
forces and copied contact forces. Original copied-host force differs from
generalized force by0.0424388N and yields4.24383e-4N*s impulse residual. Reading
actual GPU EFC indices gives3.70017e-6N force discrepancy and1.96586e-8N*s
impulse residual. In all five differing contact records of that short probe,
normal_difference + host_adhesion is exactly0. This isolates the label error;
it is not evidence of an object aid or a change to the plant integrator.

Earlier inactive/noncontiguous-constraint hypotheses were checked, not assumed:
the differing short-probe contacts have valid contiguous addresses and zero
inactive-object-contact counts. The demonstrated cause is the adhesion field.
The new reader nevertheless explicitly supports inactive/noncontiguous indices
and rejects invalid/capacity indices. No installed dependency was patched.

`src/supermimic_research/mechanics/warp_contact_labels.py` now derives local
6D cone force from actual GPU EFC indices/forces, with explicit dimensions and
friction coefficients. Controlled evaluation rejects non-joint actuation and
records no-adhesion semantics. It saves source EFC values/index ranges plus
legacy copied forces/adhesion separately, so correction is independently
reviewable. State/PD/EMA/reference/solver/gravity/material code is unchanged.
This copied-host defect does not establish a problem in native-only mug labels,
where native forward dynamics initializes contact fields itself.

## Fixed v4 repeat and remaining limits

`runs/controlled_evaluation_v4/report.json` repeats exactly the v0.59 cohort:
same four cases, seeds, frozen weights, finite clocks, initialization semantics
and gates. Full initial integration state and captured GPU physical parameters
match within each controller pair and across v3/v4. All four task cases still
fail the same gates: cube executes2s but does not track/clear penetration gates;
pen stops at0.16s against its planned7.36s. No improved policy result is claimed.

`runs/controlled_evaluation_review_v2/report.json`:6/6 saved state/control/raw-
metric checks pass. The independent additional EFC review is deliberately kept
separate: `runs/controlled_warp_receipt_review_v1/report.json` passes27/28,
not all checks. All1222 source contact rows reconstruct exactly; saved GPU geom
friction/priority rules independently reproduce the reader's local forces.
These tasks have no explicit material pairs or interval material randomization.
All four cases' force/torque sums agree with actual generalized constraint force
within3.701e-6N/6.24e-8N*m.32 copied-host rows differ; every finite normal offset
is explained exactly by the copied adhesion value. No source-index gate fails.

One predeclared5e-7N*s free-object impulse gate fails for the cube saved Actor:
maximum7.89118e-7N*s, at tick170. The remaining discrepancy is primarily the
measured acceleration force-balance remainder (~7.89e-5N), rather than contact
decoding or velocity integration. The other three impulse checks pass (cube
ZERO1.97e-8; pen ZERO3.98e-9; pen Actor2.78e-9N*s). No threshold was relaxed.
Further numerical accounting remains before treating this as a fully qualified
contact-dynamics dataset or WM-gradient certificate.

Across fresh v3/v4 runs, exact full trajectory replay is not established even
though initial state, captured physical tables and gates agree: maximum cube
ZERO maximum absolute qpos/qvel differences are0.03554/1.68946 over the2s
trajectory. The Actor differences are smaller; pen ZERO is byte-exact. qpos
mixes joint radians, translation metres and quaternion entries; qvel mixes
rad/s andm/s. These vector maxima are not pure object drift measures.
The source of this cross-run variability still needs isolation; observer-only
changes must not be presented as a cause of improved/worse physical motion.
No GPU finite-difference or determinism guarantee follows from paired starts.

## Provenance, validation and next phase

All earlier source/report snapshots remain under provenance, including the
v3 reader and short-probe source revisions. Original raw run folders are not
rewritten. The v3 raw-review6/6 result only reproduced its copied-host force
labels; it did not validate those labels as GPU force truth. Its geometric,
duration and kinematic failure evidence remains, while affected force labels
must not be used as trusted supervision without correction/reacquisition.

152 tests pass, including a native contact regression: deliberately corrupting
the copied host adhesion changes the host force, while actual EFC-derived force
stays unchanged. Other tests cover inactive contacts, noncontiguous pyramid
edges and overflow. No training optimizer step, new learned checkpoint, system
driver/CUDA change, or historical environment mutation occurred. The standing
first-healthy-neural-checkpoint restore/ETA pause condition is not triggered.

Next preserve and resolve pen initialization/contact feasibility, distinguish
static support from full original-clock tracking, and bound GPU numerical
repeatability/remaining force-balance error before neural fitting or action
gradient claims. Original contact-frame support targets, matched residual A1/A2,
independent RGB/full/multiple trajectories/total amortized costs and hardware
subversion/observation contracts remain required. Failed prefixes are not experts.
