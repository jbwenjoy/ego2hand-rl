# TRAIN-only derivative units and weight — 2026-11-16

Eight registered arms completed2000 updates each. Derivative RMS normalization
improves the observed forward/object/wrench tradeoff at the same outer weight,
but Fn6 action gradients remain unreliable. No Actor optimization or main64
policy/video promotion is admitted.

## Same-bank fixed-final development results

All arms share3866 valid TRAIN point rows,949 pair targets,initial NN,seeds,
point/pair streams,forward coefficient1,auxiliary.1 and physical masks. RAW
uses point y_scale/weight.1;RMS01 uses TRAIN derivative RMS/weight.1;RMS1 uses
the SAME derivative RMS/weight1. No Fn6 response term was added. Forward is
normalized MSE on1935 valid development rows;AD is pooled relative L2 on frozen
physical-screen pairs(object248,wrench266,Fn284). Zero-response AD baseline=1.
Reported metrics always use the original point y_scale,not the new loss units.

| Arm | Forward object | Forward wrench | AD object | AD wrench | AD Fn6 |
|---|---:|---:|---:|---:|---:|
| POINT_DIRECT |.95762|.17402|.99847|1.00016|.99889|
| POINT_IMPULSE |4.63743|.11658|.99989|1.00272|1.00515|
| RAW_DIRECT |13.72240|3.06545|.60868|.74432|1.04200|
| RAW_IMPULSE |8.69422|2.04860|.83559|1.06141|1.04869|
| RMS01_DIRECT |2.45937|1.49181|.56760|.71454|1.02979|
| RMS01_IMPULSE |6.40998|.14808|.70732|.86252|1.04114|
| RMS1_DIRECT |4.11759|1.75552|.49509|.71193|1.00235|
| RMS1_IMPULSE |14.02707|.99527|.83329|.76579|1.00338|

At weight.1,DIRECT object forward MSE13.72->2.46 while object AD.609->.568
and wrench AD.744->.715. Increasing normalized weight.1->1 improves DIRECT
object AD to.495 but worsens object forward2.46->4.12. Thus a tradeoff remains;
do not choose one favorable column as whole-model success. RMS01_DIRECT still
has worse object forward MSE than POINT_DIRECT. Fn AD remains above1 for every
response arm;no model passes all local contact/response tests. Candidate Fn
counts are RMS01_DIRECT27/284 and RMS1_DIRECT45/284,not broad reliability.

These are paired controlled comparisons on already-inspected NEVER_FIT
development histories of one original reference/fixed actor seed. They are not
unseen-original-reference/final-test/ego or multi-seed generalization results.

## Unit preparation and mechanism checks

For each of90 physical output columns,scale=max(sqrt(mean(TRAIN finite contrast
squared)),.001*point_y_scale per radian). Loss still reads only object12 plus
source-BODY moving-COM wrench6. In these18 columns,response/point scale ratios
range.216824–29.3836,median6.52620;none hit the positive floor. Raw RMS/floor/
final arrays are retained. No target centering or validation normalization.

Negative copies perturb ALL validation and invalid TRAIN x/y/heads;rebuilt TRAIN
units remain bitwise identical. Preparation4/4 passes. Each loader rederives
frozen point/pair products and response scales,without rebinding any native
source receipt. Pair and unit/mode/weight/hash identities are serialized;wrong
cache identities reject before RNG or optimizer work. RMS is resident/frozen
on GPU;point/head normalizers and networks are unchanged.

TRAIN diagnostics show RMS01_DIRECT object/wrench AD.54552/.48366 versus
RAW.36859/.40992:TRAIN objective response fit worsens while development response
improves,so this is not uniform domination. Late weighted response LOSS fractions
are RAW93–97%,RMS01~68–76%,RMS1~85–89%. These are loss values,not measured
gradient fractions or proof of a sole failure cause. All results are retained.

## Reproducibility and cost

First/final checkpoints187/187 pass NN/Adam/sampler/RNG/prediction/units and
actual next-update clone parity. Health uses TRAIN128;100/500/1000 curves TRAIN
only,development validation only at final2000. Current unit suite401/401,
evaluation980/980,TRAIN diagnosis11/11 and saved-array/unit/mask/cursor review
1186/1186 pass. Arithmetic/health success does not replace gradient admission.

Fresh POINT and RAW controls reproduce previous POINT and RANK20 models at100
and2000 exactly for NN,Adam,point/pair samplers,gradient snapshots and logs:
eight checkpoint-control comparisons/48 field comparisons,all exact. Historical
global RNG is NOT imported or claimed identical between experiments with
different numbers of initialized arms. This is a fresh replay audit,not resume.

- `runs/pen_response_unit_scales_v1`:2 point/head normalizer computations and
  2 response-unit computations,including a negative copy;0 NN/native updates.
- `runs/pen_response_unit_fit_v1`:16000 live+32 clone QA=16032 optimizer steps,
  0 native/Actor. Fit+QA87.2082s,runner90.1269s;first checkpoint estimated79.568s
  remaining compute.335 device-wide telemetry samples retained.
- The separate dry run performs one point/head and response-unit reconstruction;
  actual fit loader performs one each too. These CPU checks are not optimizer
  updates and are not hidden as pure GPU timing.
- `runs/pen_response_unit_evaluation_v1`:all480 groups/475 valid,1920 source/
  direction/model JVPs and all predictions;0 optimizer/native.
- `runs/pen_response_unit_diagnosis_v1`:all949 TRAIN pair predictions/JVPs,
  loss accounting;0 optimizer/native.
- `runs/pen_response_unit_review_v1`:independent arithmetic and control replay;
  0 new optimizer/native. Status:NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION.
- Exact ledger/sources/tests:`provenance/pen_response_unit_results`.

Six response arms add128000 endpoint forwards each;point arms do not. Equal
updates/point draws are not equal FLOPs. Previous51008 native collection steps
are not new work here. No package/driver/CUDA changes. Full physical task,
contact efficacy,genuine A1/A2,multitrajectory/cost,ego assets and hardware goals
remain active;old masks and324 CHECK are immutable.

Next isolate explicit Fn6 response supervision at existing normalized weights,
keeping object/wrench coefficients,forward anchor,scales,data and initialization
fixed. Existing Fn6 point supervision has not qualified its action gradients.
Do not replace those tests with forward loss or promote unqualified models.
