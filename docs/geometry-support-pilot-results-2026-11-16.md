# Geometry/support pilot and independent failure diagnosis — 2026-11-16

The first registered geometry-aware residual/support pilot is complete. Both
arms use the same calibrated initial state/modules/normalizers and each receives
4096 training transitions,32768 native world-integrations and64 Adam steps.
Fresh W1 mean executions terminate after.24s in both arms, versus.20s for the
common initialization. Neither completes the original7.36s task. There is no
established support benefit, physical-WM-gradient or ego-generalization result.

## Implemented contract

The preregistration is `geometry-support-pilot-protocol-v1.md` and
`configs/pen_geometry_support_pilot_v1.toml`, frozen before optimization.
`SupportPenRuntime` extends the shared owned native runtime; `SupportPenPPO`
reuses the shared collector, PPO/GAE, networks and raw action contract. The mean
head starts with zero weights and initialized-raw bias, std.05 bounded[.02,.2].
No latent offset, physics force, geometry shrinking or reference retiming is
introduced. Both arms compute all contact/support fields; support weight alone
is0 versus.2. Online data/reset paths can diverge and are not identical datasets.

Geometry uses maxima over all8 native samples: object2mm,self.2mm,joint.001rad,
nonhand.1N and finite/capacity flags. Termination is at the20ms control boundary,
not mid-integration. The original50mm/shaft60deg runtime termination remains;
strict evaluation still uses COM5mm RMS/10mm max and fullSO3 5deg RMS/10deg max.

The nominal mean contact target is computed from reference linear/angular
momentum on the original clock with actual mass/inertia/gravity. USED phase k
means interval[k-1,k], index k-1. Last3 intervals are invalid, not extrapolated.
FP64 canonical wrist/object composition avoids world-origin derivative noise.
Measured hand wrench is averaged in source BODY axes about the MOVING COM,
window slot5. This differs from fixed-source-COM slot7. The target is a simulator
prior, not measured ego force, and changes reward only.

Public checkpoint schema is `pen_geometry_support_PPO_training:3`, containing
complete native4/cache/contact/task state, networks, Adam, normalizers, rollout
and counters. A private structural adapter reuses training2 validation after
public schema/arm/immutable-target checks; public old schemas still reject.
The common-source fork changes only arm metadata/support-weight scalar.

## Verified evidence

| Evidence | Result | Scope |
|---|---:|---|
| Final GPU qualification v2 |74/74|Common fork, USED interval, source axes/moving COM, independently recomputed reward, no optimizer|
| Actual two-arm training |5/5|Equal registered total budgets; every collector25/25 before optimization|
| Fresh complete-state review |214/214|Byte/scalar/module/Adam/norm/rollout readback; reject old schema,wrong arm,corrupt target without changes|
| Actual loaded continuation |included above|Each arm4 more controls and32 diagnostic Adam steps, save/reload; not pilot candidates|
| Independent saved-pose/EFC/wrench review |114/114|Original-clock fullSO3, raw active-constraint decoding, geometry, signed WORLD/moving-COM mean wrench, support loss|
| Offline zero-wrench signal audit |10/10|Saved data only; no simulation/training|
| Full unit suite |329/329|Seven new analytic objective/phase/frame/schema tests plus prior322|

The first qualifier v1 reported72/72. Its overall no-optimizer expression used
contact-row count and was uninformative; v2 directly checks both saved optimizer
counters and empty optimizer state. Both runs actually perform zero updates.
Retain v1/source bytes rather than silently rewriting its check.

Roots in `runs/`: `pen_support_qualification_v1`, `pen_support_qualification_v2`,
`pen_support_pilot_v1`, `pen_support_pilot_state_review_v1`,
`pen_support_pilot_preview_v1`, `pen_support_pilot_preview_v2`,
`pen_support_pilot_preview_review_v1`, `pen_support_signal_audit_v1`.
Preview v1 is the video already shown to the user. v2 repeats the same declared
three cases while retaining raw EFC/contact packets and per-control windows.
Both remain authoritative for their own executions; small GPU future differences
are preserved, not called deterministic replay. Neither uses a W32 native-state
transfer into W1. Only actor/norm weights transfer; no post-start reset.

Canonical ledger: `provenance/pen_support_pilot_results_v2/ledger.json`. The first
archive included its own still-open console log; independent revalidation
detected that final printing changed its bytes. Keep that first archive. v2
excludes live archive consoles, retains the completed v1 log separately, and
recovers its exact formerly pinned prefix by original full size/SHA. Its explicit
historical alias verifies every first-archive manifest entry without rewriting
old artifacts or substituting a basename/hash-only lookup.

## Strict task outcome and ordering

Independent v2 results, all gates unchanged:

| Fresh mean case |Duration|COM RMS|Full angle RMS|Max object depth|Strict pass|
|---|---:|---:|---:|---:|---|
|Common initial|.20s|24.913mm|76.025deg|1.976676mm|no|
|Geometry residual,2 iterations|.24s|about21.54mm|about70.13deg|1.976514mm|no|
|Geometry+support,2 iterations|.24s|21.478mm|70.058deg|1.977045mm|no|

Both trained cases first exceed full-angle10deg at.1225s, then COM10mm at.145s.
The first low hand-normal-load sample is.1275s for residual and.125s for support.
These are descriptive event times, not proof of causality or loss of all contact.
Original initialization exceeds angle at.0775s and COM at.1025s. The shared
first.20s prefix has COM RMS12.412/12.383mm and full-angle RMS45.010/45.037deg
for residual/support. Both remain far outside strict gates. Shorter duration
metrics must not be compared as if all cases completed the same task horizon.

For these mean executions, object/self/joint/nonhand geometry gates, no-force,
finite and capacity gates pass; position/full-angle/duration gates fail. Training
rollouts do include geometry violations: e.g. support iteration2 max2.109mm.
Do not generalize the mean-execution geometry pass to all sampled training data.
Video is actual saved2.5ms poses,20x slow, with explicit terminal still frames.
Rendering adds no physical integrations; perfect pose is still not ego input.

## Support score distinguishes weak contact poorly in some intervals

The v1 support term is exp(-.5L), with mean squared force/.3N and torque/.003Nm.
Independent raw-EFC folding matches its stored WORLD/moving-COM hand wrench and
source-axis mean, including signs and force moments. Thus the following is a
property of the declared objective, not evidence of a wrong slot/sign/clock:

* Across365 valid reference intervals, counterfactual ZERO hand wrench scores
  have median.27253,mean.37256,max.85181;100/365(27.397%) exceed.8 and151/365
  (41.370%) exceed.5. Isotropic squared-error norms are rotation invariant.
* The INITIAL execution at control end.10s has mean hand normal.03974N and
  support score.82639; at.12s all8 hand normal samples are0 and score.76133.
* At.14s residual/support mean hand normals are.02946/.01454N yet scores are
  .82075/.77816. These mean-load rows are weak contact, not necessarily all-zero
  contact at each native sample.
* In the four training rollouts,331–399 eligible rows have mean hand load<=.1N;
  their average score is.595–.612. About19–26% of those weak-load rows score>.8.
  Even the average eligible score minus its own zero-wrench baseline is negative
  (approximately-.091 to-.101) for both arms/iterations.

The score is bounded and smooth, but a positive value alone does not certify
physical support. This observation does NOT establish that it caused the failed
policy, or that a revised reward will solve the task. Before more support
training, test preregistered contact eligibility and/or zero-wrench-centered
scores on shared saved data, then qualify a new explicitly versioned learner.
Do not retrofit v1 rewards/checkpoints or promote failed physical WM gradients.

## Cost accounting and remaining work

New cohort total:76320 native world-integrations =4096 qualifier v1 +4096 v2
+65536 registered training +272 preview v1 +272 preview v2 +2048 diagnostic
loaded continuation. Registered pilot updates128(64 per arm); additional
diagnostic updates64; total actual Adam steps192. Offline reviews/unit tests/
rendering add0 native steps and0 optimizer steps. Prior evidence costs remain
separate. This is a single-seed,single-clip pilot, not an A1/A2 multi-trajectory
or acceleration experiment. No installed package/driver/CUDA change.

The full research objective remains incomplete: robust original cube/pen task,
fair multi-trajectory/multi-seed comparison, physical WM forward/action-gradient
qualification, contact/dynamics uncertainty, real ego geometry/frame integration
and hardware observation/subversion/latency/gain validation all remain governed
by the existing plan. This turn does not create a new pause request.
