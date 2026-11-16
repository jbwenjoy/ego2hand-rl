# Response units and weight isolation — study4 protocol v1

Registered before new scale preparation/fitting. Previous study3 showed rank20
object/wrench response improvement with worse forward prediction and unreliable
Fn6 gradients. Late response LOSS fractions93–97% motivate this experiment;
they do not prove gradient domination or a sole cause. The development
validation has already been inspected and remains NEVER_FIT,not a final test.

Keep exact full20 TRAIN point bank3866 valid rows,949 eligible signed-repeat
pair targets,physical masks,source/PD/body/COM conventions,initial network7411,
point sampler7411,pair sampler7751,width512 and original DIRECT/IMPULSE families.
No new native samples,Actor updates,architecture,Fn6 response loss or forward
anchor changes. Same maintained contrast trainer/update/checkpoint/evaluator.

Before fitting compute each physical output's derivative-target RMS over ALL
949 eligible TRAIN pairs. Scale=max(RMS, .001*point_y_scale per radian),using a
one-radian reference solely for the positive floor. No mean subtraction,epsilon
rescaling or validation labels. Save raw RMS,floor,final90 scales and exact
source pins. The response objective still reads only object12 and wrench6;
scales in other columns do not introduce new supervised blocks. Perturb all
validation and invalid TRAIN x/y/heads in a negative copy,rebuild TRAIN pairs/
scales,and require scale products bitwise unchanged. Retain the old point/head
normalizers. Report the two computations and loader recomputation separately.

Eight fresh arms:POINT_DIRECT/IMPULSE;RAW_DIRECT/IMPULSE;RMS01_DIRECT/IMPULSE;
RMS1_DIRECT/IMPULSE. Point arms weight0. RAW keeps point-y-scale and weight.1.
RMS01 uses derivative RMS and weight.1;RAW vs RMS01 isolates units at equal
weight. RMS1 uses identical derivative RMS and weight1;RMS01 vs RMS1 isolates
weight at fixed units. Forward loss coefficient1,auxiliary.1 unchanged. Every
arm gets identical full20 point draws and2000 updates. All six response arms
get32 pairs/64 endpoint forwards per update,with the same949-pair bank/streams.
Point batch256,AdamW lr3e-4,decay1e-5,clip1. Equal updates are not equal FLOPs.

Serialize both pair-bank and response-unit/weight/scale identities;wrong cache
or scale resume rejects before sampling/optimizer work. Scales stay GPU resident
and frozen. Save initial NN and checkpoints100/2000. At both require finite
NN/Adam/gradients,exact NN/Adam/point+pair samplers/RNG/units reload and actual
clone-next-update parity. Total16000 live+32 clone-only QA=16032 optimizer steps.
Continue the resumed goal after healthy100;the historical pause was served.

Health uses first128 TRAIN rows;100/500/1000 curves TRAIN only. Evaluate the
inspected NEVER_FIT validation only at final2000,report ALL eight arms without
selection/early stopping. Keep all480 groups/475 valid and original frozen
physical amplitude/spread masks;point y_scale still normalizes REPORTED
forward/AD metrics across every arm. Derivative loss units must not change
evaluation units. AD candidate relative<1/cosine>=.9 remains exploratory;Fn6/
wrench/forward/full20 reliability is required before imagined Actor use.

Independently compare fresh POINT and RAW controls' NN/Adam/samplers/logs to
study3 POINT and RANK20 at100/2000. This is a replay audit,not a native/network/
optimizer resume;retain mismatches if any and do not require success by
rewriting old checkpoints. Full old snapshots stay authoritative. Report loss
fractions as loss values,not gradient fractions;include native/preparation/
QA/evaluation/point/pair/GPU/wall costs. New validation remains developmental;
unseen original references and independent trained seeds are still reserved.

No main64 policy/video promotion,ego/generalization/amortized-speed claim or
installed-package/driver/CUDA change is authorized by this fit. Original full
task,contact efficacy,genuine A1/A2,multitrajectory/cost/ego/hardware goals remain.
