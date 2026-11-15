# Contact-rich four-arm world-model fit — 2026-11-16

All four registered fresh models complete2000 live updates each;checkpoint100
and2000 health87/87 both pass. Training/QA39.467s,total runner42.086s. This is
8000 live+16 clone-only QA optimizer steps,zero Actor updates/native simulation.
No model qualifies for Actor optimization. Response supervision improves TRAIN
directional responses but worsens forward CHECK prediction;the registered
contact/gradient gates remain failed. The current policy/video is unchanged.

## Shared implementation and explicit dataset/model schemas

`learning/pen_contrast_fit.py` remains the single four-arm loop. Explicit study/
data schema2 reads the already frozen common808-valid-TRAIN/324-CHECK/188-pair
data. It deliberately constructs fresh model-family schema1 DIRECT/IMPULSE
arms;model schema2 elsewhere means DIRECT_VS_DIRECT_CV and is not reinterpreted.
The CLI takes an explicit common-data report. Historical schema1 retains its
original source/data/frozen-baseline requirements;historical artifacts are not
retrained,rewritten or silently resumed. New CPU guards reject family/CHECK/
budget reinterpretations. Full suite362/362 passes after the final core edit.

The loader reconstructs and checks valid-TRAIN-only scales,pairs and masks
against the frozen arrays. No old network,optimizer or native state transfers.
Four models share saved initial NN weights,point sampler7411 and normalizers;
response models share pair sampler7751. All808 valid point rows and188 eligible
pair groups stay GPU resident. Original8 rejected TRAIN rows and all prior
solver/epoch failures remain unchanged. Source snapshots freeze executed current
code;old full-native states retain exact-source guards,not implicit migrations.

Each model gets512000 point draws. Each response model additionally draws64000
pairs and evaluates128000 extra endpoints (256000 across both response arms);this is
equal updates/point draws,NOT equal FLOPs or wall work. No new data collection
or Actor training occurs. Loader recomputes normalization once for validation,
in addition to the earlier preparation's canonical/negative-control computations.

Checkpoint100 saves then checks exact NN/Adam/point+pair sampler/RNG/predictions,
finite parameters/gradients/logs,frozen buffers and actual next-update clone
parity. Health87/87 estimates remaining four-model compute34.79s;the healthy
run continues under the resumed goal to2000. Final health87/87 repeats those
checks. Eight clone-only QA updates per boundary are reported separately from
each live2000 cursor. A historical first-checkpoint pause was served/resumed;
no new pause or result selection occurs.

## Fixed-final CHECK evidence:negative overall result

Evaluation uses the unchanged324 inspected developmental CHECK rows,48 physical
finite-response pairs and their FROZEN earlier physical screens. New normalizers
come only from TRAIN. Stable-screen counts are36 object,34 wrench,38 Fn6;these
are local finite-amplitude screens,not certified infinitesimal derivatives.
Do not compare new normalized scores directly with older models using different
training normalizers. All four final2000 outputs are retained,no CHECK selection.

|Final model|CHECK object12 normalized MSE|CHECK wrench6 MSE|CHECK Fn6 MSE|Screened object AD relative error|Screened wrench AD error|Screened Fn6 AD error|
|---|---:|---:|---:|---:|---:|---:|
|POINT_DIRECT|1.27080|.148998|1.21465|.989530|1.00270|1.26191|
|POINT_IMPULSE|13.0847|.144064|6.35095|.988770|1.00393|1.31701|
|RESPONSE_DIRECT|15.8716|2.77368|4.05769|.928984|1.01441|1.20050|
|RESPONSE_IMPULSE|141.086|33.2831|19.4576|1.27800|1.12114|1.03373|

Zero action-response relative error is1. RESPONSE_DIRECT's object directional
error is modestly lower than POINT_DIRECT,but its forward object/wrench/Fn6
errors are much worse and contact gradients fail. IMPULSE response is worse on
object and wrench. Neither improving one block nor training/checkpoint health
demonstrates a useful controller or establishes algorithm superiority.

Candidate local AD screens(relative error<1 AND cosine>=.9,plus frozen physical
screen) pass object/wrench/Fn6 pairs as follows:POINT_DIRECT0/0/4,POINT_IMPULSE
2/0/4,RESPONSE_DIRECT4/4/2,RESPONSE_IMPULSE0/2/10. Denominators36/34/38. No model
passes all local screens;none qualifies a full20D Jacobian. Model finite-response
and AD errors are numerically close on CHECK,so these results do not indicate
a large AD-versus-model-FD implementation discrepancy. They still disagree
with physical responses. Do not use these models for imagined Actor updates.

|Check|Result|Meaning|
|---|---:|---|
|Final forward/JVP evaluation|108/108|Complete fixed-final arrays,shared scales,source/repeat/mask/finiteness contract|
|Post-hoc TRAIN/scale diagnosis|7/7|Finite records and declared loss accounting;not model-gradient qualification|
|Independent saved arithmetic|152/152|CHECK forward,TRAIN/CHECK FD/AD aggregates,original masks,Adam cursors and8016-step accounting|
|Full unit suite|362/362|Prior360 plus explicit data-vs-model schema and role/budget guards|

## What the TRAIN and excitation diagnostics actually show

TRAIN finite/AD responses use all188 eligible pairs (two original repeats per
sign). Aggregate object/wrench AD errors are POINT_DIRECT.9981/.9893,
POINT_IMPULSE.9959/.9977,RESPONSE_DIRECT.5683/.5800 and RESPONSE_IMPULSE.6271/.6662.
The response objective does teach part of the observed TRAIN action effect.
TRAIN Fn6 AD error worsens to1.0883/1.0328 for response models;Fn6 response was
not a supervised contrast objective and must not be presented as qualified.
POINT_DIRECT TRAIN object MSE.005736 versus CHECK1.2708 also shows substantial
within-reference state/response generalization difficulty despite low point loss.

The declared within-source intervention matrix has rank4 in20 physical PD
dimensions. The two CHECK sine directions have only9.663%/15.118% squared norm
inside that span. Actual TRAIN PD deltas' max orthogonal remainder is1.56e-7rad,
consistent with quantization/control tolerances rather than designed independent
excitation. This does not mean the global action matrix has rank4;cross-state
variation can constrain a model under structural assumptions. It does mean the
experiment cannot identify arbitrary local20D action effects without such
assumptions. This is a coverage diagnostic,not proof of one causal failure.

Over the final100 updates,weighted response loss is98.753%/98.687% of total
loss for RESPONSE_DIRECT/IMPULSE. These are LOSS fractions,not measured gradient
fractions. Raw initial contrast loss252.65 falls to132.98/202.04 at that window;
point+auxiliary fitting is traded off. Output-standard-deviation normalization
does not automatically balance a derivative loss's units/magnitude. Future
TRAIN-only derivative normalization/weight ablations need separate registration;
do not tune against this inspected CHECK and call it a final test.

## Timing,GPU telemetry and limits

The recorded152 device-wide nvidia-smi samples have utilization median22%,range
0–23%;device memory used1865–2100MiB of32607MiB(about5.7–6.4%). No telemetry
errors. These include process/desktop memory and checkpoint/QA periods;they are
not model-only occupancy,torch peak allocation or ordinary PPO measurements.
The small four serial MLP updates remain underutilized. No GPU speedup claim:
this turn changes data-contract support,not launches,batches or installed code.
Timing excludes earlier8800-step collection and common preparation;the reported
total runner additionally includes this invocation's loading/cache/setup. These
are component costs,not complete multi-reference SuperMimic amortization results.

Next profile the ordinary32-world collector/PPO path to address the user's
geometry/synchronization concern with real call counts and matched clean timing.
Then register fuller local20D excitation and isolated derivative-loss scale/
forward-anchor comparisons using TRAIN and new reserved validation data. Keep
all physical/geometry/no-aid gates and independent CHECK labels;do not add more
updates or select a checkpoint simply to turn this registered result green.

Exact receipts/data/checkpoints/logs live in the unified repository's ignored
`runs/`;compact evidence is `provenance/pen_contact_response_fit_results`. The
old CLI source referenced by historical tracked loss reports is recovered from
Git80053c3 with its original Windows bytes,and only accepted after matching its
original full-path/size/SHA pin. Existing archives are not altered. The shared
maintained implementation remains in `src/`,not another study directory.

Full controlled task success,genuine model-based A1/A2,multitrajectory/multiseed
cost benefits,real ego object/geometry/dynamics/frame admission and hardware
contracts remain incomplete. No new policy/video result is claimed.
