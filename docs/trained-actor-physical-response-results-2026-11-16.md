# Trained-actor contact responses and frozen WM gradients — 2026-11-16

The new source bank/probes are complete:12 declared sources,324 geometry-legal
20ms counterfactual controls,3776 native world-integrations and0 neural updates.
Independent final packet/plant/frame checks6525/6525,screen arithmetic586/586,
restore profile50/50 and337 unit tests pass. All48 plus-minus pairs exceed their
observed3-repeat variation;36 object-motion pairs also pass the declared two-
amplitude screen. Nevertheless all three old WM families fail to predict useful
object action responses on this cohort. These records are CHECK/NEVER_FIT.

## Actual sources and input contract

The frozen protocol/config are `pen-trained-response-protocol-v1.md` and
`pen_trained_response_probe_v1.toml`. Use final64-iteration residual and centered-
support actors,not selected intermediate checkpoints. Fresh W1 mean prefixes
capture reference phases10,20,30,40,60,75(time.18,.38,.58,.78,1.18,1.48s).
Both entire source banks finish BEFORE any counterfactual restore. Producer
prefixes use no post-start reset or restoration. These are new diagnostic
executions,not claimed bitwise copies of the old video.

All12 sources satisfy declared nonterminal/finite/no-aid/capacity/previous8-
sample geometry/nonhand gates and mean hand normal>.1N. This is physical/contact
eligibility,NOT full reference tracking qualification or expert status. Later
sources can already violate the strict orientation tracking gate.

Complete source containers include native4/Model/Data/cache/metric/controller/
RNG,immutable support-table contract,source observations,actor checkpoint and
actual mean raw action. A separate W1 counterfactual runtime restores each full
source BEFORE every branch. Arrays/scalars and current observations read back
exactly. Byte equality is not promoted to deterministic future physics.

Each source has3 baseline repeats and2 fixed unit sine directions x2 PD amplitudes
(.002/.005rad) x2 signs x3 repeats=27 branches. Source raw u becomes u+4sign*eps*d;
actual unchanged scale.5/EMA.5 gives the intended PD perturbation only without
clamps. Actual transmitted PD is checked and held8 native samples. No clipping
repair,body force,reference retiming or expert relabeling. All324 are available
and pass online geometry and independent physical quality; no replacement source.

The maintained recorder/restore helper is `runtime/pen_owned_probe_recorder.py`.
It observes output externally and never becomes hidden physical runtime state.
Independent decoding reuses maintained `warp_packet_decode`, `controlled_pen_records`
and state55/warm26/actualPD20/nativeDt1 input with90-output H20 chart. Actual
engine warmstart is recorded; acceleration is not substituted for that memory.

## Physical and frame checks

All raw active EFC indices/forces,signs,friction/generalized force/torque,linear
impulse,source qpos/qvel/warm,clock,PD,mocap and no-aid checks pass. Maximum
residuals over every recorded native tick are5.562e-7N(force),1.192e-8Nm(torque),
1.413e-7Ns(linear impulse),below unchanged1e-4N/1e-5Nm/5e-7Ns gates. Geometry
remains object<=2mm,self<=.2mm,joint<=.001rad,nonhand<=.1N. None of these green
record checks certify a learned gradient or successful full task.

The audit explicitly compares cached xmat/xipos with post-integration source
qpos BODY/COM. At these12 CONTROL BOUNDARIES the rotation difference is only
1.07e-8..4.35e-8rad and COM difference0. This does NOT confirm the suspected
stale-frame bug; no source-frame fix is justified by these measurements. Keep
native-step pre-integration contact epoch separate from the control-boundary
source epoch. New WM labels independently rotate mean WORLD/moving-COM wrench by
actual source qpos BODY; resident cached-axis labels also match their own frame.
Future performance changes must preserve this observed boundary relationship.

All404 actual GPU Model arrays and every scalar leaf match the old WM training
parent's fixed model at dt.0025 for every source. Matching is stronger than equal
XML names or mass/inertia alone. Model schemas and physics dictionaries match.
The sources follow trained actors and later states; this is a distribution-
shift diagnostic,not a guarantee of old training-state coverage.

## Frozen models still fail physical response

The initial review v1 tests DIRECT and DIRECT_CV and passes6522/6522 record
checks. Explicit extension v2,registered before querying the additional model,
adds the predetermined candidate-IMPULSE final2000 checkpoint. Its prepared
dataset and frozen y scales match,and its duplicate DIRECT model weights equal
the DIRECT in the CV comparison exactly. Both prior comparisons had2000-update
budgets. Preserve separate checkpoint pins and v1/source bytes; do not portray
the extension as an originally preregistered three-family study.

All models remain read-only. JVP is w.r.t ACTUAL PD along the2 declared directions,
not autograd of a measured-wrench reward or the raw action tensor. Compare it
with repeated native central finite responses divided by2epsilon,using the
frozen TRAIN y scales. Repeats are numerical executions of fixed inputs,not
independent trained seeds or probabilistic confidence bounds.

|Object12 normalized AD relative L2|All48 qualified pairs|36 amplitude-consistent pairs|
|---|---:|---:|
|Zero action-response comparator|1|1|
|DIRECT|1.8803|2.2692|
|DIRECT_CV|1.8646|2.4249|
|IMPULSE|3.5008|4.4763|

The stable subset was predeclared: both amplitude contrasts exceed observed
spread,FD(.002) vsFD(.005) relative difference<=.25 and cosine>=.9. It does not
prove a smooth contact derivative. All48 pairs exceed repeat spread for object,
wrench and Fn6;stable subset sizes are36,34,38 respectively. A larger observed
effect is not sufficient for differentiability or model accuracy.

For the stable subsets,wrench6 AD relative errors are1.6711/1.4571/1.6674 and
Fn6 errors.9087/.8659/.9983 for DIRECT/CV/IMPULSE. A few individual directions
meet the candidate model screen,including2/48 object pairs for CV and0 for the
other families. Do not qualify an entire model/Jacobian from those few pairs.
Object response remains worse than the zero-response comparator even after
excluding amplitude-inconsistent pairs. Physical coupling alone did not solve
this new-distribution response problem.

|Frozen forward normalized MSE on324 valid rows|DIRECT|CV|IMPULSE|
|---|---:|---:|---:|
|Object12|3.0643|4.1015|17.1755|
|Source BODY COM wrench6|3.0478|4.7076|3.3806|
|Fn6|7.2775|7.8410|8.7043|

The data demonstrate measurable action effects on legal contact states,not
uncontrollability. They also do not establish why a given WM fails. Coverage,
forward errors and local response losses must be investigated on SEPARATE TRAIN
sources; never train on this CHECK cohort or silently turn it into an untouched
final test after using it for diagnosis. No model-based Actor update is made.

## Restore path profile

After one warmup per source,36 instrumented full restores take17.722s,mean.4923s.
All byte/scalar checks pass and native/optimizer work is0. cProfile cumulative
times include17.419s in native.restore,15.348s in preflight,11.570s across26172
verify_file_pin calls,and8.861s in36 verify_source_snapshot calls. Disk reads and
Windows path resolution are prominent. These nested times OVERLAP and must not
be added. Instrumentation affects timing;there is no uninstrumented speedup or
GPU-saturation claim.

This identifies repeated provenance I/O in the DIAGNOSTIC RESTORE path,not the
ordinary PPO collector bottleneck. Investigate an explicit verified-source
validation epoch or prepared restore transaction,with default strict loading
unchanged and source/task/corruption/byte-equality guards preserved. Do not just
remove validations or claim this fixes the earlier training utilization issue.

## Artifacts, costs and next work

Roots under `runs/`: `pen_trained_response_probe_v1`,
`pen_trained_response_review_v1`, `pen_trained_response_review_v2`,
`pen_trained_response_screen_summary_v1`, `pen_trained_source_restore_profile_v1`.
Tools: `collect_trained_pen_responses.py`, `review_trained_pen_responses.py`,
`summarize_trained_pen_response_screens.py`, `profile_owned_source_restore.py`.
Checkpoints/source states/raw packets/predictions/profile remain outside Git;
the exact ledger and compact evidence are in `provenance/trained_pen_response_results`.

New physical cost3776=1184 uninterrupted producer-prefix steps +2592 branch
steps. All reviews,the coupled-model extension,screen arithmetic,restore profile
and337 unit tests add0 native steps/optimizer updates. No new network,Actor,
package,driver or CUDA change. Earlier full-task failures remain unchanged;
this round does not produce another successful-task or ego video.

Full research objective ACTIVE/incomplete. Next priorities are separately
declared contact-rich TRAIN coverage/shared data and response-aware model tests,
with this CHECK cohort never fitted and a separate final holdout for any later
claim. Keep original cube/pen gates,multitrajectory/multiseed A1/A2 and hardware/
ego contracts explicit. Readback passes and finite-response data do not certify
CHORD physical action gradients or permission to optimize a model-based Actor.
