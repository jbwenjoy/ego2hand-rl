# Matched rank4/rank20 world-model fit — 2026-11-16

All six registered arms completed2000 updates from the same initial network,
full20 TRAIN point bank,point streams,valid-TRAIN scales and update budget.
Rank20 response supervision improves some held-history object/wrench action
responses,but damages forward prediction and does not fix Fn6 gradients.
No model qualifies for imagined Actor optimization;main64 policies/video stay.

## Actual fixed-final reserved-validation results

Forward columns are normalized MSE on1935 valid validation rows. AD columns
are pooled relative L2 error on frozen physical-screen pairs (object248,
wrench266,Fn284). Zero-response comparator is1. Different measures must not
be compared directly. All480 declared groups,including5 physically masked
groups,remain in the saved evaluation.

| Arm | Forward object12 | Forward wrench6 | AD object12 | AD wrench6 | AD Fn6 |
|---|---:|---:|---:|---:|---:|
| POINT_DIRECT |0.95762|0.17402|0.99847|1.00016|0.99889|
| POINT_IMPULSE |4.63743|0.11658|0.99989|1.00272|1.00515|
| RANK4_DIRECT |5.47328|0.56565|0.90804|1.14047|1.04169|
| RANK4_IMPULSE |8.49214|2.74220|0.99822|1.35053|1.03568|
| RANK20_DIRECT |13.72240|3.06545|0.60868|0.74432|1.04200|
| RANK20_IMPULSE |8.69422|2.04860|0.83559|1.06141|1.04869|

RANK20_DIRECT passes exploratory local candidate tests for167/248 object,
161/266 wrench,and17/284 Fn pairs. It does not pass all local screens,let alone
a reliable full20D contact Jacobian. Some gradients better than zero are not
enough for control. Its forward Fn6 MSE0.22654 is lower than POINT_DIRECT0.35789,
yet Fn action gradients remain worse than zero:forward loss alone is inadequate.

The rank4 and rank20 response arms use the SAME full20 point samples and equal
point/pair draws. This isolates additional supervised response directions within
that point bank;it does not attribute changes versus the old816-row bank solely
to rank or establish a sample-collection/amortized speed advantage.

## TRAIN diagnosis and remaining hypothesis

Across all949 eligible TRAIN pairs,RANK20_DIRECT object/wrench AD errors are
0.36859/0.40992,versus RANK4_DIRECT0.93491/1.01083. The trained model learned
additional responses;forward/gradient tradeoffs and validation gaps remain.
All20 validation directions lie in the requested full20 span. Their projection
fractions onto the four TRAIN supervised directions range~.000886–.416894.
Physical masks still leave source ranks17–20;do not call every state full20.

The response term contributes~95.13–96.90% of late loss in rank4 arms,and
~93.18–97.17% in rank20 arms. These are LOSS VALUE fractions,not gradient
fractions or proof of a sole failure cause. Point-y-scale-normalized derivatives
have different units/magnitudes from point targets. This motivates a separate
TRAIN-only response-scale/weight experiment,with unchanged forward anchor,
data/initialization/budgets. Do not silently add Fn6 response supervision or
change several losses and attribute the result to normalization alone.

## Integration and validation isolation

Maintained pen_contrast_fit/update implement study3;model-family schema remains1
DIRECT/IMPULSE. Historical study1/2 and DIRECT_VS_DIRECT_CV meanings remain
separate. Pair-bank/subset identities are serialized;wrong identities reject
before RNG sampling or optimizer work. All rank arms share resident point data.
The existing evaluation/diagnosis/review tools now explicitly accept study3.

First and final checkpoints each pass135/135 checks:NN/Adam/point+pair samplers/
RNG/predictions/gradients/cursors/frozen buffers and actual clone-next-update
parity. Health probes use the first128 valid TRAIN rows. Curves100/500/1000
contain TRAIN only;reserved validation first evaluates at fixed2000. No labels
enter fit/scales/early stopping/selection. Original inspected324 CHECK is unused.
The new validation is NOW INSPECTED developmental NEVER_FIT,not a fresh final
test for future experiments. Whole-original-reference tests remain reserved.

Full unit suite395/395 passes. Final evaluation976/976,TRAIN diagnosis9/9 and
independent saved-array/cursor/pair-identity/mask/curve review1125/1125 pass.
These arithmetic/health passes do not erase failed physical/gradient criteria.
Independent review status:NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION.

## Artifacts and cost

- `runs/pen_fullrank_fit_v1`:six x2000 live optimizer updates,plus24 clone-only
  QA updates=12024;0 native/Actor steps. Checkpoints100/2000,initial NN,curves,
  all six outputs and254 device-wide GPU telemetry samples retained.
- Training+QA wall66.1964s;runner70.5812s includes loader/source/setup costs.
  The first healthy checkpoint estimated57.465s remaining compute. No historical
  native state/network/Adam is resumed. Loader rederives TRAIN-only products
  once,verifying frozen data rather than rebinding native source receipts.
- `runs/pen_fullrank_evaluation_v1`:all final predictions and1440 source/
  direction/model JVPs,all480 finite groups;0 optimizer/native steps.
- `runs/pen_fullrank_diagnosis_v1`:all949 TRAIN finite responses/JVPs and loss
  accounting;0 optimizer/native steps.
- `runs/pen_fullrank_fit_review_v1`:independent saved-array arithmetic;0 work.
- Exact sources/tests/pins/cost ledger:`provenance/pen_fullrank_fit_results`.

Response arms perform128000 extra endpoint forwards each,point arms do not;
equal updates/point draws are not equal FLOPs or wall cost. GPU telemetry is
device-wide,not process occupancy or a general speedup result. Previous51008
native data-collection steps are not charged as new steps here. No installed
packages/drivers/CUDA changes. Full task,contact efficacy,genuine A1/A2,
multitrajectory/seeds,total cost,ego assets and hardware goals remain active.
