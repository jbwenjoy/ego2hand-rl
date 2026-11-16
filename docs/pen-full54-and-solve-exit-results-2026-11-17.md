# Closed full54 evaluation and actual solve-exit observations

All original M0–M5/true A1/A2 requirements remain open where not demonstrated.
This checkpoint closes two bounded experiments, not the research goal.
All paths below are relative to the consolidated repository.

## Full54: no full-task success

`runs/pen_native_screened_rollout_v1/report.json` closes all54 cases,8/8 checks,
68,840 actual=attempted native integrations, zero optimizer/normalizer updates,
14,835.3268s wall time. The final independent V3 report passes5/5 and reviews
all54 in registered order; `indices/completed_54.json` preserves its fixed index.
The producer's root work.json remains a stale53-case progress record. It is not
a live-job indicator or final result and is excluded from the evidence pins.

`runs/pen_native_screened_summary_54_v1/report.json` passes7/7:

- 0/54 completed the7.36s reference;34 BASE_PHYSICAL_INVALID and20 TASK_FAILURE.
- 8,605 queries,4,565 selected controls,2,506 accepted proposals, including7
  accepted duplicate-BASE proposals. Duplicate SHAM acceptance is retained.
- 5,574 restores and13,170 source captures. Recorded restore/capture time sums
  are5,462.1758s/8,214.7085s. Categories overlap; these are not isolated speed
  measurements or an acceleration claim.
- RESPONSE_ONLY same-source OBS/FULL RAW gradients are identical over1,023
  inspected sources. Its positive constant path-head value is not a useful
  action derivative. Distinct executed trajectories can still differ.

The independent decoded-packet taxonomy at
`runs/pen_screened_failure_taxonomy_54_v1/report.json` passes17,266/17,266:
8,542 valid queries and63 invalid,34 BASE+29 ALT. All63 violate only the original
EFC linear-impulse gate5e-7Ns. Their peak residual min/median/p95/max is
5.098020587e-7/1.051407180e-6/2.366704938e-6/2.866139584e-6Ns.
All63 have terminated=false,truncated=false. Physical stopping censors future
outcomes: the34 stops do not establish object fall, eventual task failure or
eventual success. No gate was relaxed.

Both fixed-repeat0 nine-panel videos at `runs/pen_native_screened_video_v1/`
pass10/10 checks. They display saved actual2.5ms poses,20x slow, no interpolation
or new dynamics. Six decoded start/mid/end frames were visually inspected.
The aggregate outcome and diagnostic figures were also visually inspected;
labels/legends are legible. These are controlled native simulations, not ego
retargeting, hardware execution, world-model animations or newly trained Actors.

## Actual V2 observations: complete120-query/960-tick budget

Use only the actual28-wide V2 context implementation described in v1.49.
Both fresh native processes A/B finished60 queries/480 integrations each,
6/6 producer checks, zero setup advances/updates. Wall times154.7396s/154.2187s
include audit/storage and are not a solver performance benchmark.
Each separate saved-data reader passes1,266/1,266, recomputing35,280 array
elements with zero native steps. Source restores, exact held PD and actual
copied epochs were checked. The V2 registration/actual worker/reader binding
attestations and thin-entry snapshots remain part of the evidence.

The all120 summary at `runs/pen_solve_exit_summary_v2/report.json` passes7/7.
It preserves every condition, both scopes and all masks. Original/new validity:

| Historical valid | Newly valid | Queries |
|---|---|---:|
| false | false |36|
| false | true |24|
| true | false |12|
| true | true |48|

Only3/5 historical failing conditions recur: SUPPORT_P058, SHAM_P064_ALT,
RANDOM_P067_BASE. UNMODIFIED_P027_BASE and SHAM_P110_BASE no longer fail.
Historical passing SHAM_P064_BASE now fails; it shares source and RAW with
SHAM_P064_ALT. All ten conditions have12 queries each;3 numerical repeats,
2 fresh processes and2 scopes are not independent training seeds or population
incidence estimates. Only46/120 complete traces match their historical bytes.

Of the60 observed queries,24 fail the original impulse gate. At every failed
peak, recorded improvement=0, with2–4 iterations, gradient/newton-decrement
criteria false and iteration-cap50 not reached. The actual scaled solver
gradient term's signed projection on the full original impulse residual is
0.9994926459–0.9999537543; Ma accumulation projects only
-1.724736482e-5–1.169979171e-7. This localizes these observed residuals to the
nonzero solver-gradient term and identifies improvement-only early termination
as a candidate mechanism. It does not establish why improvement is zero
(precision, line search or stagnation), uniquely explain all historical cases,
or prove that changing termination will improve a task.

The observed gradient is the native acceleration-cost/generalized-force
residual, not an action gradient. It provides no WM action-derivative or Actor
admission.

## Observer differences remain explicit

`runs/pen_solve_exit_scope_pairs_v1/report.json` directly reloads all60 paired
raw traces, matches every stored field difference and passes5/5 coverage
checks. The first implementation assumed a fixed scope order; the actual
counterbalanced schedule rejected it before output/native work. Revision
7c840c8 handles either order before the successful saved-data run.

Only49/60 pairs are fully byte-identical. All60 have equal impulse masks,
physical masks and task flags; actions, initial recorded state/warmstart,
held PD, GPU time and mount poses are identical. Max differences include
post-qpos4.768371582e-7 (mixed coordinates), post-qvel5.390942097e-4
(mixed velocities), qacc/warmstart0.2157669067 (mixed accelerations),
generalized constraint force2.665817738e-5 (mixed force/torque), reward
2.038478851e-5 and peak original impulse5.647523974e-10Ns.
Unmatched raw contact/EFC row differences are not aggregate force errors.
Equality of masks in this sample does not prove observer neutrality or future
determinism; the data do not isolate the observer as the sole cause.

## Consequence for the next experiment

Do not rerun full54 or train on these inspected diagnostic traces. First make
an explicit bounded solver-termination comparator with its own implementation
identity, source snapshots and registration. Compare original OR termination
against a candidate that removes only improvement-only exit while preserving
gradient/decrement tests, max50 and every numerical value/physical gate.
Use the same ten sources/actions, both observation scopes, two processes and
three numerical repeats:240 queries/1,920 integrations total, zero updates.
Preserve nonconvergence and partial failures; compare all masks, residuals,
iterations, state/force changes and complete costs, not only rescued cases.
The native installed environment, driver/CUDA and old results stay unchanged.
Candidate physical transitions need a new backend identity and downstream
requalification before any training or task-benefit claim. If identity/source
transfer cannot be made exact, stop before native execution and version the
implementation rather than silently reusing the old backend registration.

True A1/A2, independent seeds/references, fitting split ownership, full-task
improvement, GPU training performance, real ego asset/gravity/dynamics/RGB
validation and hardware observation/control/latency contracts remain open.
