# Batched GPU physical oracle before support-aware retargeting

Previous goal turn made progress: fc69031 preserves the four failed actual-GPU
pen runs, independent 76/76 and EFC 28/28 review, 159 tests and current videos.
The full Goal remains active. This phase performs no neural optimization,
physical integration, trajectory reset or expert-data fitting.

Build a resident forward-only oracle from the saved projected pen's host
topology plus all 15 actual GPU physical tables. Restore full fixed wrist,
object pose and original source velocities. Explicitly address the 20 named
hinges/actuators; do not assume the object-last state layout. Every query has
zero solver warmstart and independent joint positions/PD targets. Validate
the all-joint, free-object, zero-COM-offset, no-equality/no-aid contracts.

The object contact wrench is read from actual generalized constraint force:
world translation force rotated into BODY and local rotational force at COM.
Validate this shortcut independently using actual EFC contact decoding and
contact-point moment arms. Keep actual contact penetration and nonhand/self
contact diagnostics; these do not replace a final all-pair native convex check.

First compare restored static GPU snapshot against its saved actual GPU
contacts/forces, tolerance 1e-4 N and 1e-5 Nm. Test one moving-source query and
an 81-world batch: center and plus/minus 1e-3 rad perturbations of 20 joint
positions and 20 PD targets. Compare each result with the same individual
queries, using the same force/torque tolerances. Save qpos, ctrl, qvel, qacc,
constraint forces and decoded contact receipts for readback. Failures stay
failed; do not relax thresholds or call a differing reconstruction identical.

Capture only the repeated forward workload in a CUDA graph after warmup;
fixed models/state/tables stay on GPU and queries do not allocate an env.
One aggregate device-to-host readback per batch is allowed for optimization;
do not insert per-contact or per-world host checks inside the forward loop.
Benchmark three identical 81-query sweeps as batched versus individual after
warmup, separating setup/compilation and reporting exact query budget. This is
a physics-query benchmark, not neural training or end-to-end retargeting speed.

Before any nonlinear search, check finite-difference behavior with a declared
direction, h=1e-3 and h/2. Report actual discrepancies and mode changes. Such
finite differences are local physics diagnostics, not a validated learned
world-model action gradient. If batch/replay fidelity fails, repair or explain
that bridge before using the oracle to propose initial states.

Subsequent support-aware search must keep original nominal dynamic wrench,
source velocity/clock, legal original actuator control/torque ranges, bounded
joint/residual changes and independent geometric/execution gates. A local
low-wrench residual alone cannot qualify a full trajectory. Static support,
derivative uncertainty and moving-hand compliance remain separate requirements.
Keep first-healthy-neural-checkpoint restore/ETA pause for future training.
