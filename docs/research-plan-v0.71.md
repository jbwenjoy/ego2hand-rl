# Fresh controlled pen world-model data and first paired checkpoint

Previous turn was progress4dcd2be: explicit device tolerance resolves a
host/device policy mismatch; strict8 actual cases pass56/56 force/momentum
checks but all tasks fail.169 tests pass. Full original Goal remains active.

Acquire24 FRESH actual no-aid/no-reset episodes:both dt, TRAIN action seeds
7201..7208 and CHECK7301..7304. Environment seed/selected initial pose/clip
stay fixed. Independent local PCG action plans add AR(0.5) Gaussian sigma0.04
raw noise clipped to+/-0.08 about original2*delta, every20ms for the entire
original clock until first native termination. No favorable case selection.
Use the explicitly versioned strict device numerical profile. All actual
failed prefixes remain genuine nonexpert dynamics, not successful examples.
Do not reuse prior CHECK/diagnostic artifacts as TRAIN.

Pack20ms transitions into existing state55/world-left-delta52 semantics,
with EXPLICIT named joint/free-object indices (object FIRST in this model).
Inputs102:state55,actual engine warmstart26,executed PD target20,native dt1.
Outputs90:delta52,next warm26,mean source-BODY COM wrench6,mean Fn6 for
five fingers plus palm/other robot. Integrate WORLD contact wrenches about
instantaneous COM over all actual physics ticks, then rotate by SOURCE R.
Never average vectors in varying BODY frames or assume the first20 qpos
are hand joints. Fn6 includes all robot-object contact groups; nonhand support
is separately flagged. Use actual mass/inertia/grav priors, not measured ego
parameters. Mean horizon is exactly20ms at both physical dt.

Retain all rows and predeclare a numerical-quality mask requiring every tick
to pass original EFC force1e-4N/torque1e-5Nm/linear impulse5e-7N*s, finite/
capacity/no-applied-force and zero nonhand support. Task/geometry failure
flags are retained separately: real failure dynamics can train the WM, but
are never expert imitation labels. Fit scales from valid TRAIN only. Require
at least64 valid TRAIN and16 CHECK rows, source/frame/codec validation and
observed impulse-defect closure before starting any optimizer.

Reuse the maintained Compound/ImpulseResponse architecture/decoder via a
separate90-output pen schema (five fingers+other robot), preserving old87-output
mug source/checkpoints. Matched DIRECT versus IMPULSE heads, same width512,
NN weights, valid TRAIN rows, batch256, samples and100/2000 update budget;
coupled arm predicts measured WORLD numerical impulse defects rather than
inventing contact forces. Both supervise actual measured BODY wrench/Fn.
This isolates a representation/auxiliary package, not Actor A1/A2 success.

If data checks pass, first saved checkpoint after100 paired updates. At that
boundary STOP training, verify exact model/Adam/scales/RNG/sampler readback,
finite losses/gradients/heads and cloned next-update parity. Estimate remaining
1900 paired compute updates from measured CUDA timing, separating setup/QA.
Then PAUSE the Goal under the user's standing instruction. No continuation
to2000, Actor fitting, full-task claim or checkpoint-accuracy claim by numerical
health alone. Physical action-gradient/response and rollout accuracy remain
required after resumption, as do full controlled learning, matched support
objectives, ego/multi-trajectory total costs and hardware contracts.
