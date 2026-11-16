# Full20 controlled physical-response data — 2026-11-16

Both registered collectors completed. All36 sources passed source gates;all5832
declared branches were retained and passed ONLINE geometry checks. Independent
EFC/native/PD/frame/geometry decoding nevertheless rejects31 rows on the
unchanged5e-7Ns linear-impulse gate. No threshold or physics was relaxed.

| Cohort | Sources | Rows | Physically qualified | Eligible signed-repeat groups |
|---|---:|---:|---:|---:|
| TRAIN |24|3888|3866|949/960|
| Reserved validation |12|1944|1935|475/480|

Rejected TRAIN indices:1194,1274,1315,1359,1363,1397,1441,1445,1661,1743,1802,
1882,3249,3255,3287,3331,3337,3369,3741,3785,3823,3867.
Rejected validation indices:663,853,865,881,935,947,963,1324,1404.
Each failed row remains in NPZ and metadata with its exact gates;pair eligibility
requires both repeats of both signs to pass. The packet-quality flag also
reflects these impulse failures;it is not a second independent failure cause.

## What coverage is actually established

Requested PD directions are orthonormal20,condition1. After physical masks,
the intersection across both amplitudes retains ranks17–20 per source:
18/24 TRAIN and9/12 validation sources retain20. Actual executed PD matrices
are also checked per fixed amplitude,with minimum rank17. Tiny FP32 residue
across amplitudes is not counted as extra rank. Full-Model equality across
all36 sources checks404 arrays and195 scalars,not selected physical fields.

Data review7312/7312 passes source/role/plant/actual-PD/mask bookkeeping checks.
This does NOT mean all raw physical rows pass or all directions are smooth.
Using frozen OLD valid-TRAIN descriptive scales,the existing two-amplitude/
two-repeat screens retain the following directional conditions:

| Block | TRAIN (of480 directions) | Validation (of240 directions) |
|---|---:|---:|
| object12 |277|124|
| source-BODY moving-COM wrench6 |338|133|
| Fn6 |382|142|

These screens require both physical groups,relative amplitude error<=.25,
cosine>=.9,and contrast exceeding the observed repeat envelope at both
amplitudes. Observed envelopes are not confidence bounds or infinitesimal
derivative proofs. Different blocks can retain different directions. No world
model was evaluated or qualified by these data-only screens.

## Matched future training products

New maintained `learning/pen_fullrank_dataset.py` consumes only explicit
preparation2 TRAIN and VALIDATE_NEVER_FIT. It does not relabel old CHECK.
One common full20 point bank feeds both rank4/rank20 supervised-response
comparisons. Eligible TRAIN pair counts are189 (directions0..3) and949
(directions0..19). The point bank and its valid mask stay identical;this
isolates response-supervision direction coverage,not sample-collection cost.

Fresh point/head scales use3866 valid TRAIN rows only. An independent negative
copy changes ALL validation x/y/heads and all invalid TRAIN x/y/heads;TRAIN
scales,heads,pairs,subsets and audits remain bitwise identical. Common products
pass9/9 checks. Two normalizer computations include this negative check;zero
NN/Actor/native updates occur in preparation. Full test suite390/390 includes
role relabelling,mixed validation,group overrides,actual-PD corruption,matched
subsets and invalid-label leakage checks. No installed packages changed.

The six-arm rank-only fit protocol is registered BEFORE fitting/evaluating any
model on reserved validation. It keeps the same model families,initial NN,
point bank/streams,valid-TRAIN scales,updates and original response scale/weight.
Adapting the maintained trainer to this explicit study3 contract is NEXT;
the new config is a frozen planned study,not a completed or yet-executed fit.
Derivative-target scaling/weights/forward anchors are separate future ablations.

## Exact artifacts and costs

- `runs/pen_response_fullrank_train_v1`:33376 native integrations,0 optimizer.
- `runs/pen_response_fullrank_validation_v1`:17632 native,0 optimizer.
- Corresponding `*_preparation_v1`:81/81 and51/51 bookkeeping checks,all
  exclusions retained,0 native/optimizer.
- `runs/pen_fullrank_data_review_v1`:7312/7312,0 native/optimizer/normalizer fit.
- `runs/pen_fullrank_common_data_v1`:9/9,0 native/optimizer,2 scale computations.
- Exact sources,tests,source relocations and final ledger:
  `provenance/pen_fullrank_data_results`.

Total cohort51008 native integrations INCLUDES4352 producer integrations already
recorded by the earlier running registration. Increment since that producer
registration is46656;do not add the full total to it twice. Both collectors
used distinct processes/output trees and overlapped on the same hardware;
this is not a clean throughput experiment. All sessions are now terminal.

These are independently noisy histories of the SAME original reference and
fixed trained actors,not independently trained seeds,unseen original references,
final-test or ego data. Old324 CHECK and816 TRAIN remain immutable. Main64
policy/video unchanged;old WM contact/action gradients still fail admission.
Full task,multitrajectory total-cost,ego assets and hardware goals remain active.
