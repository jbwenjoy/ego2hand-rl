# Registered centered support training trial v1

Freeze this file/config/source and require a passing offline score comparison
before optimizing. It selected CENTERED using preregistered analytic invariants,
not policy improvement. Start two NEW shared-initialization learners; do not
continue or rewrite v1 absolute-score weights/rewards. New public schema is
`pen_geometry_support_PPO_training:4`; old training3 must reject cross-loading.
Historical sources/checkpoints remain archived; changed live source bytes are
not silently accepted as a historical native continuation.

All v1 reference/clock/assets/physics,161/188 observations,total raw Gaussian
actions/clamp/scale/EMA/PD,source BODY axes/MOVING COM,USED interval k-1 and last3
invalid targets remain. Original geometry and full task gates are unchanged.
No body aids,teleports,post-start evaluation resets or reference retiming.

The geometry/base reward is unchanged. Support term alone becomes
exp(-.5L(measured,target))-exp(-.5L(0,target)) on the same valid/geometry/native
eligibility mask. It is signed and bounded[-1,1]. Force scale.3N,torque.003Nm,
support weight.2; no extra contact-load threshold or contact observation added.
Zero net wrench gives0,including balanced squeeze. This does not bound normal
squeeze; all geometry gates stay enforced. Centering is not potential-based
policy-invariant shaping and can alter termination incentives. Report terminations
and common prefixes; never infer task success from a positive score.

Both arms share one freshly calibrated native4 source, all actor/critic/norm/
optimizer tensors and RNG before fork. Mean-head weights0/bias initial raw,
std.05 clamped[.02,.2], seed7411. Each arm64 iterations x64 controls x32 worlds
=131072 transitions,1048576 native world-integrations,2048 Adam steps.
Both total2097152 native/4096 Adam. Shared8-control calibration2048 native;
qualification two4-control cohorts cost2048 more. Count extra diagnostics,
failures,evaluation and retries separately. Online rollouts may diverge.

Save a complete checkpoint at iterations1,8,16,...,64 and health/consistency for
every iteration. Store rollout memory separately only at checkpoint boundaries;
each complete checkpoint already contains that memory. Never claim unsaved
iterations have disk checkpoints. Report remaining per-arm and total loop ETA
at the first healthy checkpoint. A previous first-checkpoint pause was already
served/resumed; this protocol does not create another pause request.

Fresh W1 mean evaluations: common initialization and each arm's final64-iteration
actor/norm only. Same original full7.36s reference or first native/geometry
termination,zero post-start reset. Save raw EFC/contact/pose and all failures.
Independent original-clock fullSO3/geometry/wrench readback and complete state
restore/diagnostic continuation remain separate from registered training cost.
This single-seed/clip controlled perfect-pose trial does not establish ego
generalization,multiseed/multitrajectory A1/A2,physical WM action gradients or
speedup. All of those remain explicit unfinished research requirements.
