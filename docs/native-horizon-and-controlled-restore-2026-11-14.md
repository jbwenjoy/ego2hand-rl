# Native horizon results and controlled pen restore

The previous goal turn was progress: the three-column actual replay was rendered,
decoded and visually checked, committed as `4cd5fc4`. The H6 process finished
normally. This turn independently reviewed its saved results and exercised the
original pen checkpoint through the unified source tree. Full Goal remains active.

## Fixed H6 results

`runs/mug_native_horizon6_v1/report.json` is the executed, preregistered v0.57
experiment. Same initial state, reference, action bounds, EMA, physics and gates
as H3. Only REFERENCE/NATIVE_MPC at the two original integration steps was run.
The longer summed residual window changes the objective relative to once-per-
action regularizers and increases cost; it is not a matched-compute comparison.

| Integration step | H3 duration | H6 duration | H6 COM RMS | H6 angle RMS | H6 maximum self penetration | Median decision wall time |
|---|---:|---:|---:|---:|---:|---:|
| 0.3125 ms | 0.868750 s | 1.328750 s | 22.058 mm | 20.478 degrees | 0.318500 mm | 1.4420 s |
| 0.15625 ms | 0.901563 s | 0.960469 s | 16.116 mm | 13.282 degrees | 0.210054 mm | 2.7498 s |

Both H6 cases abort at the unchanged 45-degree orientation envelope. Both fail
the original 0.2 mm self-penetration gate and the 2 s tracking qualification.
There is no stable-hold/retargeting success or dt robustness result. The coarse
duration gain is larger than the fine gain; it must not be presented alone as
a generally solved control improvement. Reference COM RMS acceptance is 5 mm.

Total wall time 246.981904 s; 1,969,920 oracle integration steps and 10,399 active
execution steps. Oracle counts include repeated k0/5 decision checks, exclude
the separately recorded earlier parity audit. Thirty full-horizon branches per
decision remain. About 24.25%/23.47% of active action coordinates are within 0.01
of the normalized box boundary. Median local merit reductions are only
2.73%/3.22%. These observations do not prove control-authority infeasibility.

`runs/mug_native_horizon6_review_v1/report.json`: 7/7 applicable independent
readback checks pass, reconstructing 53,091 raw contacts, controls, COM/velocity,
tracking metrics and optimizer/cost accounting. Historical-zero parity is
explicitly N/A: there are no new ZERO cases, and the pinned H3 baseline review
remains the source for those traces. 114 complete selected first-interval native
states equal actual execution exactly; all four repeated decisions match.

`runs/mug_native_horizon6_chain_v1/report.json`: 36/36 supplementary checks pass,
covering finite arrays, k/tick/duration/partial-terminal clocks, full native and
observed state/warmstart carry, per-tick pose/velocity carry, zero initial EMA,
actual EMA/control updates, initial reference pose and zero velocity. These
checks complement the raw contact review; they do not turn failed tasks into
successful demonstrations. Warning counts still rely on recorded online counters,
and there is no new per-state convex geometry audit.

`runs/mug_horizon_video_v1/zero_60ms_120ms_actual.mp4` displays the fixed coarse
ZERO/H3/H6 trajectories at 4x slow motion, with explicit terminal stills and
separate physical clocks. It has 209 decoded 1536x800 frames at 30 fps. It is
actual saved qpos playback, not predicted-state rollout or a learned Actor.

## Original controlled scope: what is and is not established

The historical `dex-supermimic-research/outputs/windows_bringup_20261101` records
show cube short updates 0->12, cube separate exact-parameter resume 12->16, and
pen short updates 0->12. Actor/WM/critic changed in all three. These use play
configuration, zero contact-head loss and disabled failure head; their 100-step
simulation checks manually reset done environments. Pen reports three terminated
transitions. They establish pipeline bringup, not no-reset physical tracking or
an experimental baseline faithful to the complete original training objective.
The pen recipe selects the single-axis GROUP, not a single clip.

The unified pen restore initially failed because the native loader resolved
`assets/resources/aruco_pen250` beside the new source tree, while large assets
were intentionally preserved in their original external location. Failure and
its exact tool/intent/log are preserved in
`provenance/controlled_restore_path_failure`; no historical result is overwritten.

The sole existing Dex source change is an optional explicit `xml_path` argument
in `assets/pen.py` and `get_pen_cfg`. The prior default remains. The new restore
tool passes the external XML through task configuration without patching module
globals, adding imports from historical studies, copying models into Git or
changing physical parameters. The immutable archive and original installed
environment/source are untouched. This is a declared asset-resolution change,
not an unreported exception to frozen experiment source pins. Existing H3/H6
source pins do not include this loader and remain valid.

`runs/controlled_pen_asset_equivalence_v1/report.json`: 2/2 compiled model
comparisons with the immutable archived loader are exact at mass 0.03 and
0.05 kg. All exposed arrays/options except dt are hashed; both use the same
default dt. XML and all adjacent texture bytes are pinned. This checks model
identity, not real-object calibration.

`runs/controlled_pen_restore_v2/report.json`: 24/24 checks pass on RTX 5090,
iteration12, sim_steps736. Modules, optimizer state, torque normalization, task
reference signature, counters and CPU/CUDA/NumPy/Python RNG restore exactly,
and remain exact after a newly saved checkpoint is read back. All Dex imports
come from this unified repository. Wall time31.197941 s; peak Torch allocated
memory19,149,824 B (not total GPU process/device memory). No learn/optimizer
step was invoked. This checkpoint does not contain native environment state or
replay data: it is not proof of bit-exact continuation of simulation/training.

The external pen XML/recipe are now explicit artifact-registry entries. Their
clips root is supplied explicitly. The cube's source-relative object/jig
resources have not yet been validated under the unified task loader; the
historical cube resume is not silently relabeled as a new unified validation.

## Next work and retained Goal

Do not launch another neural fit merely to learn failed mug diagnostic actions.
The next controlled milestone is unified cube/pen asset loading and a shared,
no-object-aid, no-reset evaluation with an explicitly selected single pen clip,
full pose/velocity/contact/termination clocks and initial-state semantics.
Audit command resampling/phase resets and external-wrench events before treating
existing play/reward logs as that evaluator. Existing checkpoint restoration
and actor playback are separate from successful tracking qualification.

For the mug, a later preregistered action-sequence or search comparison may
separate constant-action/local-search restrictions from authority; no larger
action limits or easier physical gates are justified by the current results.
Preserve actual dt differences and all native computation costs.

All original object-frame support, matched A1/A2, action-gradient validation,
ego geometry/inertia/material/frame/gravity/input/controller contracts,
independent RGB/full and multiple trajectories, total amortized costs and
hardware subversion/joint/zero/gain/latency/observation requirements remain.
145 repository tests pass, including corruption checks for saved chronology
and explicit-path pen geometry/inertia checks. No new training was started,
so the first-healthy-checkpoint training pause condition was not triggered.
