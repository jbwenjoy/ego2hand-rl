# Research plan v0.54: measured-wrench impulse decoding with explicit defects

2026-11-13. The previous goal turn made progress: b01d1ad committed exact
continuation and recursive diagnostics, the fixed2000 ABS/CF budget completed,
40/40 structural checks passed, and the requested pause was observed. The user
has resumed the Goal. Do not treat completed fitting as a reliable controller.

## Evidence motivating a new representation

At the end of that fixed budget, new small-action CHECK AD response NMSE versus
zero prior1 was ABS/CF:.569281/.670225 delta52,.963219/1.234178 object12,
.959031/1.182553 contact9. The proposed CF loss did not improve the final result.
At200ms, COM error was3.740/3.998mm, orientation error.063745/.077610rad.
Mean linear momentum defect RMS was5.76e-4/5.86e-4Ns versus native4.63e-8Ns;
angular defect was3.82e-6/8.91e-6Nms versus native8.77e-11Nms. These gaps must
not be hidden by low training loss or byte-exact checkpoint health.

## Decoder/label preparation already executed, no new fitting yet

`mechanics/impulse_codec.py` preserves the external current102/response87
chart. Independent pose heads remain q/dq40, COM delta3 and WORLD-left rotation
delta3. Predicted mean contact force/COM torque6 is in the frozen source BODY
axes, with force and moment kept separate. Rotate it once to world and integrate
over the actual full control horizon H=.02s:

`m*(v_next-v_source) = H*(F_world+m*g) + defect_linear_WORLD`

`I_world_next*omega_next = I_world_source*omega_source + H*tau_world + defect_angular_WORLD`

The compiled nominal mass, COM inertia (including principal-axis rotation),
gravity and lack of free-object damping/gravcomp are explicitly pinned. They
are simulation priors, not measured ego/real-hardware parameters. Orientation
is still a separate conditional prediction; momentum closure does not make
that pose or its contacts feasible. Position is not inferred from mean force.

Six defect labels are calculated from actual endpoints AND independently
measured contact wrenches: WORLD linear Ns and angular Nms. They never replace
contact labels or enter inference features. Only TRAIN fits their mean/std;
unit floors are1e-7Ns and1e-9Nms. They are not arbitrarily forced to zero.

Preparation used3258 TRAIN rows (old378+new2880),1920 read-only CHECK rows. True
heads with true nonzero defects reconstruct velocities to3.55e-15 on TRAIN and
9.99e-16 on CHECK. Omitting defects has maximum velocity-component error8.99e-5
on TRAIN,2.61e-6 on CHECK. F32 input/head quantization with true defects gives
maximum8.54e-8/2.46e-8; this is retained separately from physical model error.
All6 preparation checks pass, including unchanged measured force labels.

Frozen ABS/CF postprocessing views with zero defects were also evaluated
WITHOUT fitting. Small-action object AD NMSE was.963295/1.154803; this does not
establish useful response merely by imposing momentum closure. No model is
selected from these views; original data/CHECK roles remain fixed.

Hot algebra is pure Torch, F64 physical arithmetic with FP32 NN heads. Rodrigues
has finite derivatives at zero. Static3x3 cofactor products avoid CPU-synchronizing
linear solves and account for the small nonorthogonality from F32 source columns,
without rotation projection/repair. No native calls, geometry, scalar validation
or label encoding occurs inside updates. Boundary checks and tests own domains.

## Predeclared matched new experiment

`configs/mug_impulse_first100.toml` defines DIRECT and IMPULSE. Both take NN
weights from the PREDECLARED original single1000 WM, before the CF comparison.
Both zero the last-layer rows46:52, reset Adam, use identical new sampler seed,
batch256, learning rate.001, clipping10 and planned2000 updates. This is a NEW
representation experiment, not an exact resume of the old semantic heads or
an optimizer continuation. Trainable parameter count and initialization match.

DIRECT predicts independent delta-velocity6 in rows46:52. IMPULSE predicts
observed numerical impulse-defect6 there and decodes velocities through the
predicted wrench and inertia. All other head meaning is unchanged. Original
TRAIN input/output scales stay frozen; only IMPULSE defect-head scales are
newly fitted from TRAIN. Both receive the same decoded absolute loss plus.1
normalized auxiliary loss on their own six head targets. There is no CF loss.
Thus the tested change includes representation/decoder and corresponding
auxiliary labels; do not claim a decoder-only isolated effect.

Use3258 TRAIN rows only; all CHECK heads, probes and56 recorded-control cases
remain read-only. Fit both arms on identical sampled batches. Verify their
actual first update against one expendable restore update each, charging those
two extra clone updates separately. The first checkpoint is100, Adam100 per
arm. Save full state, buffers, physics/head scales, RNG/samplers, source/data
config pins, logs and gradients. Verify exact readback/next batch/predictions
and matching sampling cursors, then report independent native action responses
and20..200ms recursion. Estimate conditional remaining1900 paired-update compute
from the last50 CUDA events and PAUSE Goal at the first healthy checkpoint.

No optimizer update after checkpoint100 to improve/check the reported outcome.
Algebraically small momentum defect is BY CONSTRUCTION and cannot substitute
for native response, force accuracy, recurrence, geometric contact feasibility
or actual no-aid control. Record poor results without relabeling them success.

## Full objective and subsequent work retained

After explicit resume, complete the predeclared budget if structurally healthy,
then use independent physical response/recurrence evidence to guide real-native
reference-conditioned feedback/Actor exploration. Establish current observation,
EMA and future-reference contracts, real no-aid returns and matched task-level
baseline/contact-support comparisons before claiming algorithmic benefit.

The full controlled cube/pen loading/playback, short training/save-resume/no-aid
evaluation; object-frame support and matched A1/A2; contact/action gradient and
collision/inertia/material/gravity/input/controller contracts; independent RGB
ego/full trajectories; multi-trajectory shared-vs-per-trajectory TOTAL cost;
hardware subversion/joint/zero/gain/latency/observations remain required. One
nominal mug/reset and annotation-informed task are not those deliverables.
No driver/CUDA changes, duplicate workspace study roots or Goal completion claim.

Evidence: `runs/mug_paired_response_resume2000_v1/health_2000.json`,
`runs/mug_impulse_preparation_v1/report.json`. New fitting status belongs only
to `runs/mug_impulse_response_v1/health_0100.json`, if independently generated.
