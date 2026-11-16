# Explicit Fn6 response addition — study5 protocol v1

Registered before new fitting/evaluation. Study4 improved the object/wrench
tradeoff but Fn6 action gradients stayed unreliable. Test missing explicit
Fn6 response supervision without changing existing terms,physics,data or NN.

Reuse the exact full20 TRAIN point bank3866 valid rows/949 eligible pairs,
original31 physical exclusions,frozen90 derivative RMS scales from study4,
point/head normalizers,initial NN7411,point seed7411,pair seed7751. No new
statistics fit for the loss units;loader rederives old products/scales once.
The RMS includes Fn columns already,but study4 never used them in its response
loss. This study must record their explicit inclusion and any zero-signal floor.

Eight arms:RMS01_DIRECT/IMPULSE,FN01_DIRECT/IMPULSE,RMS1_DIRECT/IMPULSE,
FN1_DIRECT/IMPULSE. RMS controls retain object12+wrench6 means. FN adds the
Fn6 mean normalized finite-contrast squared error,coefficient1 within the
response sum. DO NOT divide the old terms by a new block count. Outer weights
.1 or1 match within each pair. Existing object/wrench coefficients,forward
coefficient1,auxiliary.1,network families/width512/architecture and point loss
remain unchanged. No geometry/force/torque/impulse threshold relaxation.

Every arm gets2000 updates,point batch256,pair batch32,AdamW lr3e-4/decay1e-5,
clip1. Same point and pair streams/banks. All arms now perform64 pair endpoint
forwards per update. Equal updates still do not imply identical arithmetic
cost. Report actual point/pair/native/QA/evaluation/GPU/wall costs separately.

Serialize pair identity and response-unit contract v2 with explicit block list.
Wrong blocks/units/cache reject before RNG sampling/optimizer work. Preserve
study1–4 payload meanings. Save initial NN and100/2000 checkpoints,requiring
finite NN/Adam/gradients,exact NN/Adam/point+pair RNG/units/block reload and
actual next-update clone parity.16000 live+32 clone QA=16032 optimizer updates,
0 native/Actor. Historical first-checkpoint pause was served/resumed;continue
the registered budget automatically after healthy100.

Health probes TRAIN128. Curves100/500/1000 TRAIN only;evaluate already-inspected
NEVER_FIT development validation only at2000. At100/500/1000/2000 also log
object,wrench,Fn component losses on the SAME complete949 TRAIN pair table;
unused Fn loss is diagnostic in control arms. No validation choice/early stop.
Evaluation units remain POINT y_scale for ALL arms. Preserve480 groups/475
valid and original independent physical screens;do not rescreen using new
model scores. Physical masks/ranks17–20 and zero-response=1 stay visible.

Independently compare all four fresh RMS controls at100/2000 to study4 RMS
NN/Adam/point+pair samplers/gradients/logs. Keep replay mismatches if any;never
repair old artifacts or import old NN/optimizer/native/RNG into new training.
Report all eight arms and all forward/contact/finite-response/AD outcomes.
Passing some local directions or accounting checks does not qualify Actor,
full20D contact Jacobians,complete task or ego/final-test generalization.

No model-based Actor update before the declared forward/contact/gradient
requirements are met. Main64 policy/video remains unchanged by this study.
Longer model-free control-task training and real no-aid evaluation remain a
separate necessary milestone;unqualified WMs cannot enter those updates.
No installed-package/driver/CUDA changes,external publish or original-data edits.
