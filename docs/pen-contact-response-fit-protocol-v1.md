# Contact-rich TRAIN:registered common data and next response-loss fit

Register before normalization or optimization. Keep original808 valid TRAIN
rows,8 masked impulse failures,188 qualified paired responses and324 inspected
developmental CHECK rows. Targeted solver replay data is diagnostic only and
does not enter TRAIN. One original reference,not unseen-reference generalization.

Preparation verifies full404 GPU Model arrays/scalars between all original
TRAIN/CHECK sources,roles,source snapshots and identical chart semantics. Copy
point rows with masks unchanged. Fit normalization/head scales ONLY on valid
TRAIN via the maintained fit_pen_scales;CHECK and invalid TRAIN must not affect
them. Store scales,point datasets and188 pair records with original row indices.
Each pair uses mean y across the two original same-sign repeats,finite contrast
(mean_y_plus-mean_y_minus)/(2*epsilon),exact same state/warm/dt and verified actual
PD difference2*epsilon*direction. Keep the4 excluded groups in the group audit.

The next optimization is four fresh arms POINT_DIRECT/POINT_IMPULSE and
RESPONSE_DIRECT/RESPONSE_IMPULSE,using maintained PenResponseWM,update_impulse/
update_contrast,and checkpoint/health components. Do not clone their physical
model or trainer into another research directory. Config supplies width512,
point batch256,pair batch32,AdamW lr3e-4,decay1e-5,clip1,auxiliary.1,contrast.1.
Fresh common NN weights(init7411),equal point streams(seed7411) and response
pair streams(seed7751). Family-specific output/head scales are shared between
point/response versions and fitted from the same valid TRAIN. No old trained
network or optimizer continuation under changed scales.

Use the existing response objective:normalized finite-contrast error averaged
over object12 plus source-BODY moving-COM wrench6,weight.1. Fn6 remains a point
supervised head and independent response diagnostic,not an unreported added
loss. Four arms each get exactly2000 live updates,with checkpoint100/2000 and
fixed diagnostic curves100/500/1000/2000. No CHECK selection/early stopping to
claim final generalization. Response arms use64 extra pair-endpoint forwards
per update;this is equal updates/point draws,NOT equal FLOPs or wall cost.
Report point/pair/native/preparation/QA costs and GPU telemetry separately.

Checkpoint100 must verify finite weights/Adam/gradients,exact model/optimizer/
sampler/RNG reload and actual next-update clone parity. Continue automatically
to the registered2000 budget after healthy checks under the user's resumed goal;
the historical first-checkpoint pause was already served. No new pause request.
Count clone-only QA updates separately from live updates;do not hide them.

At fixed2000,report forward object/warm/wrench/Fn6 errors and physical finite-
response/AD-JVP comparisons on all original developmental CHECK conditions.
Retain the original two-amplitude/repeat-spread screens and zero-response=1
baseline;these remain local directional diagnostics,not a full20D Jacobian.
Do not call model-based Actor optimization until its gradients qualify.
No improved policy/task/video is established by WM loss or checkpoint health.
Reserve whole-original-trajectory splits and multiple independently trained
seeds for final generalization and shared-total-cost conclusions.

This goal turn may complete common data preparation without optimizing yet.
Preparation alone must report neural_updates=0. No dependency on proving a
private solver termination cause is introduced for the already-qualified rows.
