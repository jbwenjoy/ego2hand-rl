# GPU contact label correction and reference/initialization audit

Previous goal turn was progress:3183a28 added actual original cube/single-pen
evaluation and playback with exact paired starts. Four cases failed. No new
neural training is planned here; full research Goal remains unchanged.

This turn's saved-action audit2/2 reconstructs pen original reference+.5-scale
/.5-EMA targets exactly. ZERO follows the moving reference, both cases have no
absolute target clipping, and reset joints equal frame0 exactly. Initial
reference target movement is only0.0059824rad max. Do not blame an unverified
zero-angle/open-hand mapping for the observed0.16s failure.

A bounded original GPU cube12-tick replay separated generalized force, copied
contact force and integration. The large impulse discrepancy is in the copied
normal contact label, not the measured acceleration/integrator. Actual source
EFC indices give a maximum force mismatch3.70e-6N and momentum residual1.97e-8
N*s; copied mj_contactForce gives0.04244N and4.24e-4N*s. On all five differing
contact decodes, normal_difference + host_contact.adhesion is exactly0. The
installed Warp get_data_into does not copy/initialize the new adhesion field;
MuJoCo3.11 contactForce subtracts that field. This is a copied-host label
problem, not evidence of adhesive object actuation or broken native integration.

Predeclare v4 repeat of the same four cases, same seeds/weights/clock/initial
state/physics/gates and stable TorchScript setting. Only observation/label
reading changes: use actual GPU EFC indices/forces and explicit cone algebra,
preserving inactive contacts as zero and rejecting invalid/capacity indices.
All actuators must be joint transmissions, so adhesive body actuation is
unsupported and rejected. Save source indices/EFC forces, copied-host labels
and adhesion values separately, plus generalized force/qacc receipts. No
installed library, system driver/CUDA or frozen old data is patched.

Retain v3 and previous reviewers/source snapshots unchanged with a correction
ledger. Their geometric/kinematic failure conclusions stand; their copied-host
contact-force labels are not validated as GPU force truth by merely reproducing
the same labels. Native-only mug contactForce is a separate path: this finding
does not invalidate native force reads where adhesion is actually initialized.

Independent new review must reconstruct forces from saved EFC receipts and
compare full-body generalized-force sums and object impulse, not another call
to the defective copied-host force path. Require unchanged explicit task gates
and paired initial physical states/parameter tables. Do not promote a successful
reader check to successful manipulation, learning or ego generalization.

After labels are audited, preserve and examine the pen2.882mm initial geometry
overlap, original reference/robot geometry mismatch and feasible bounded initial
correction with actual unassisted native dynamics before further fitting. Do
not silently replace the original source reference or relabel these failed
rollouts as experts. Original support targets/matched A1/A2/action gradients,
independent RGB/full/multiple trajectories/total costs and hardware contracts
remain pending; first healthy neural checkpoint restore/ETA pause remains.

Primary source for net-normal adhesion subtraction:
[MuJoCo3.11 engine_core_util.c](https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_core_util.c),
lines1001–1020. The installed Warp IO implementation and all local evidence
are pinned in the resulting provenance. Public sources do not replace local
runtime reproduction.
