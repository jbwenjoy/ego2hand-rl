# Controlled cube/single-pen no-reset evaluation preflight

2026-11-14. Previous goal turn was progress, committed ca54864: H6 raw/chain
reviews confirm failures, and original pen checkpoint restoration passes24/24.
Full Goal unchanged. This protocol does not start neural training.

The original cube0->12->16 and pen0->12 smoke policies are not trained experts.
Evaluate their saved Actor and ZERO separately, one environment, seed42, with
the same initialization within each task. Pin checkpoint bytes; no selection
among checkpoints/clips/seeds. Pen selects exactly 01_single_axis/clip_001
(369 frames at50Hz), not the ten-clip group. Run its original finite horizon
(368 intervals/7.36s) or first termination; cube uses one fixed goal for2s.

Use original play physics and sensor interfaces, not its easier acceptance
rules. Pen first-frame hold is disabled, uniform single-clip start0; restore
original50mm position/60degree shaft termination, with no time stretching.
Original reset writes reference frame0 and its derivative velocity once.
Command compute advances time_steps to1 at reset; the pre-action command then
denotes the next20ms target. Capture that target before env.step and align
each POST state to its actual reference time; record before/after phase to
expose any offset. Stop at reference end; never pad beyond the original clip.
Cube goal sampling remains original seeded reset; min_goal_interval is longer
than evaluation so success cannot switch that goal. Keep initial COM as the
cube's position reference. No pose injection after reset.

auto_reset=False. Do not clear manual-reset-pending flags; stop at the FIRST
native terminated/truncated result. There is one explicit reset per case.
Keep only reset/startup deterministic size events, disallow step/interval
events. Fixed wrist/mocap pose must remain unchanged. Require no object force,
no applied generalized/body force, no gravity compensation, no equality aid.
Robot PD actions and original task EMA/limits are unchanged.

Save every physics substep through an observational wrapper around the original
GPU sim.step; do not reintegrate/reconstruct CPU contact truth. MuJoCo-Warp's
installed get_data_into copies actual GPU contact/constraint outputs into a
separate host MjData. Read forces with mj_contactForce there. PRE contact/COM
and POST qpos/qvel stay distinguished; wrapper does not call mj_forward/step.
Record applied forces, mocap, actuator forces, qpos/qvel/ctrl, raw contacts and
capacity counters. Diagnostic host synchronization costs are charged; this is
not a training-performance or GPU speedup benchmark.

Report common full-object position/orientation errors; pen shaft error/native
task flags remain separate from full6D reference tracking. Use declared5mm/10mm
COM RMS/max,5/10degree rotation RMS/max,2mm native penetration,0.2mm self,
1mrad joint excursion and0.1N object-nonhand contact thresholds. Completion,
finite dynamics, no constraint/contact overflow and all physical gates are
required. Unknown GPU warning counters are disclosed, not read as native zero.
Tail contact coverage and velocities are descriptive at this stage; do not
claim the original mug's tripod/static hold gate or dynamic feasibility from
this different cube/pen protocol. Save full reference and actual clocks.

Explicit external cube XML/jig paths are additive loader arguments; move the
cube missing-default-path error from import to actual loading so an explicit
path can be supplied. Preserve defaults and compare compiled models against
immutable archive implementations. Keep all assets external/byte-pinned.
Do not mutate historical environments/archives or old reports/source pins.

Remaining full scope: successful controlled learning/no-aid evaluation,
object-frame support and matched A1/A2/contact/action gradients, true ego
assets/dynamics/frame/controller contracts and independent RGB/full/multiple
trajectories/total costs; hardware contract before deployment. Failed diagnostic
rollouts must not become expert labels. If neural training is later started,
stop at first healthy checkpoint, validate restore/ETA and pause as requested.

## Recorded preflight revisions before final receipt run

v1 completed all four cases but failed its exact pair-initialization check:
pen qpos differed by up to1.19e-7 and qvel1.26e-5. It remains a failed pairing
diagnostic, not a causal policy comparison. No acceptance tolerance was relaxed.
v2 explicitly disables TorchScript executor optimization for the WHOLE process
evaluation context; local installed torch.jit.optimized_execution implements
this switch. GPU Warp physics and Actor weights are unchanged. All v2 initial
qpos/qvel/reference pairs match exactly. This evidence is consistent with
script-transform path changes; it does not isolate or prove their precise cause.

v3 keeps v2 settings/gates and adds recorded receipts before interpretation:
full initial native integration state, host model topology, actual GPU physical
tables, pre-action observations, every termination/truncation flag and actual
GPU clock per physics substep. Pair equality now also includes the full initial
integration state. This is an explicit strengthening of reproducibility, not
relaxation of the failed v1 pairing. v1/v2 source/report/log snapshots are
preserved under provenance; their original raw run folders remain untouched.
