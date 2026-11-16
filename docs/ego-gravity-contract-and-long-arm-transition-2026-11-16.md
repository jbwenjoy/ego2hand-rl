# Ego gravity contract and long-control arm transition

## Real inputs and asset status

The three fixed-revision HOT3D pilot archives remain local. For the previously
registered right-hand cup window,clip001884 frames47–107,all61 raw timestamps,
existing body/palm transforms,reference files and compiled Hand2/cup model pins
were checked again. Prior asset9/9 and frame15/15 results remain their original
evidence;this audit does not rerun their physics or repeat a grasp experiment.

The cup's128-piece compound collision asset already exists in the simulator.
Mass0.25kg,COM/inertia from surface-lamina geometry,mass range0.15–0.5kg and
hand friction range0.3–1.0 remain explicit unmeasured priors. Collision loading
is not the missing prerequisite;source gravity binding,parameter calibration,
successful closed-loop contact and independent RGB reconstruction remain open.

HOT3D's processing description includes alignment of MPS output to OptiTrack.
Consequently MPS's own gravity convention does not certify this clip's final
annotation frame. This follows the distinction in the official references,
not an estimate of actual gravity:
[HOT3D Appendix A.3](https://arxiv.org/html/2411.19167v2#A3),
[MPS trajectory fields](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory).
The four per-frame JSON types in this61-frame window have no gravity-named
fields. That does not rule out external calibration or full-sequence metadata.
The checked local README matches the newly fetched official README bytes.

## What fixed-palm reanchoring changes

Let H(t) be the source-world-from-human-palm rotation and R the fixed robot-
world-from-robot-palm rotation. The existing kinematic mapping applies
A(t)=R H(t)^T to source-world directions. A constant source gravity g would
therefore map to A(t)g,which generally varies through the trajectory.

The audit computes a direction-independent geometric bound:

```
D(t) = A(t) - A(0)
Q = mean_t D(t)^T D(t)
min_{|u|=1} RMS_t |D(t)u| = sqrt(lambda_min(Q))
```

For these61 poses,the minimum is0.02938436655. At magnitude9.81m/s²,even the
best unit direction has RMS change0.2882606m/s² relative to its initial mapped
value. Eigenanalysis,independent stacked-matrix SVD and the minimizing direction
agree. This is a property of the recorded rotations,not gravity estimation.

Conditionally assuming source g=[0,0,-9.81],mapped gravity differs from robot
Z-down by111.908–117.768deg and varies up to4.337deg from its initial direction.
These are explicitly CONDITIONAL values;they are not a calibrated gravity
error or an explanation of past grasp failures. Do not choose a convenient
source gravity direction by fitting this window.

Even a zero bound would not establish dynamical equivalence:removing source
wrist motion changes accelerations,and a time-dependent coordinate change
also requires noninertial terms. Fixed-palm relative motion is a new nominal
robot task;source human loads/forces cannot be copied as robot truth. Explicit
simulation priors can still be used for this new task,with all original gates.

New audit8/8 passes,0 native integrations/optimizer. Four tests cover axial
gravity invariance,noncommuting positive/tight bound,world-frame equivariance,
reflection and zero-vector rejection. Full423 tests pass in9.894s. The first
test log is retained:one fixture used an incompatible SciPy Euler array shape;
the corrected fixture specifies batched rotation vectors. No physics changed.
Three primary-source snapshots are retained with URL/content pins;timestamps
carry explicit offsets (the original observed_UTC field name is historical).

Evidence:`runs/mug_gravity_transport_audit_v1/report.json`.

## Long training transition

RESIDUAL has finished its registered2000 rounds. Saved checkpoint counters:
64000 Adam updates,4096000 transitions. SUPPORT then starts from the same
common network/normalizers/empty Adam;only registered arm/support-weight
metadata differs. Its first optimized checkpoint's complete ordinary reload
has equal854 native arrays/857 scalars,NN/Adam/memory/observations/normalizers.

CPU-only milestone review14/14 independently checks the saved initial modules,
PPO config/metadata,empty Adam,final/first counters,nonempty optimizer step
counts,finite modules and the first SUPPORT reload. No physical state is
restored by that review. It adds0 native/optimizer work. The original13-check
inline report remains;the14-check version has a reusable saved review command.

Evidence:`runs/pen_long_control_second_arm_milestone_v2/report.json`.
This is a static milestone,not current liveness or final trial accounting.
The parent continues polling session89759. Final actual work remains null
until both arms terminate successfully;do not add embedded64000/32 Adam
counts to a new cohort or infer task success from training phase/reward.

Next:finish SUPPORT2000,then run the registered nine-case LIVE W1 evaluator
using `configs/pen_long_control_evaluation_v2.toml` and saved actual PD/drives.
No earlier checkpoint selection,gates,budget,physics or training source change.
All original M0–M5,real-ego dynamics,multitrajectory/seed total costs and hardware
contracts remain outstanding as applicable. Goal ACTIVE/incomplete.
