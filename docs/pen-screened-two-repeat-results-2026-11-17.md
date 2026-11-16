# Two complete numerical repetitions and registered solve-exit replay

The registered full54 study remains incomplete. Its first36 cases have all
received16/16 V3 independent saved-record checks. The immutable index is
`runs/pen_native_screened_review_v3/indices/completed_36.json`, SHA256
`c21f81fec8c27ac98d5936d8dc8f42467c0bd4df6ffa6f7b1c7cf9325f0a228e`.
The complete-repetition summary passes7/7 at
`runs/pen_native_screened_summary_36_v1/report.json`, SHA256
`b92e418079e1fce82fdca6b72216f1d2bc0f8a4c75dc90fc555266f77f034446`.

All36 cases/18 controller-teacher groups are retained in fixed declared order.
There are0/36 full7.36s successes,19 BASE physical-audit stops and17 task
failures. The remaining18 declared cases are explicitly unreviewed at this
checkpoint, not silently counted as failures. The CSV contains all54 rows and
the inspected plot shows both repetitions, every controller and stopping type:
`runs/pen_native_screened_summary_36_v1/cases.csv` and `all_outcomes.png`.
These are numerical repetitions with frozen models on one controlled reference,
not independent training seeds or real ego validation.

There are3232 selected controls,6110 native queries and48880 native integrations
in these36 closed cases. Of13248 declared controls,10016 future controls are
unavailable after stopping. All19 invalid BASE and22 invalid ALT queries remain
explicit;6069 queries pass physical qualification. Of1769 accepted proposals,
5 are duplicate-BASE SHAM proposals. The observed case wall sum10555.1649s
includes audit/storage/concurrent CPU review and prior graphics; timings do
not establish isolated compute speed or total adaptation acceleration.

The fixed original/SHAM comparisons illustrate repeat variability:

|Teacher/controller|Repetition0|Repetition1|
|---|---|---|
|RESIDUAL original|0.54s physical stop|3.50s task failure|
|RESIDUAL SHAM|2.20s physical stop|3.48s task failure|
|SUPPORT original|2.06s task failure|0.64s physical stop|
|SUPPORT SHAM|2.06s task failure|1.48s physical stop|

Physical-stop durations include the stopping control and differ from a fully
qualified prefix. Do not treat an audit stop as proof the object fell at that
time. The full all-controller plot/CSV is authoritative; this fixed null-control
table is not a ranking or selection of favorable models. RESPONSE_ONLY's
same-source OBS/FULL RAW gradient differences are exactly0 for all726 saved
source gradients across both repeats. Its differing executed lifetimes cannot
identify an explicit-path-score benefit.

`runs/pen_first_two_repeat_initial_comparison_v1/report.json` passes5/5 for
all four teacher/original-SHAM source pairs, SHA256
`3b064c55bbf122a083152c84583eae749a06e507b20ffcbce81e09efb4aa624f`.
The model, Python and RNG trees, first qpos/qvel, sampled RAW and actual PD
are equal within each pair. Native Data is not wholly byte-equal: the report
lists2/9/15/10 differing arrays respectively, including contact.geomcollisionid
and/or island index tables. Small subsequent first-control differences remain.
This is not evidence those arrays caused the differences, or proof the solver
context was equal. No full-initial-state equality or future determinism claim
is made. Failing-tick context observation remains necessary.

The replay worker is now implemented and CPU-registered, without native launch:

- Config `configs/pen_solve_exit_replay_v1.toml` and protocol
  `docs/pen-solve-exit-replay-protocol-v1.md`.
- `runs/pen_solve_exit_replay_registration_v1/registration.json`, SHA256
  `1ae9e4ce532b3320a4ecc59f13822ad5b651d50d9e746f4173e3cfed26489fac`:
  A/B fresh processes,60 queries/480 ticks each, total120/960, no updates.
- Six focused scheduling/epoch tests and586 full CPU tests pass. The full suite
  took11.827s with CUDA disabled in that test process. Epoch tests include
  distinct native/decoder gravity and reject later-forward or wrong-tick data.
- Actual worker default-entry pending guard passes5/5 before simulator import
  or replica output, at
  `runs/pen_solve_exit_replay_pending_guard_v1/report.json`, SHA256
  `a99272eae48a486f661ee64802643273e6b6c313c13bd011a6a337024a730190`.

The worker records all complete native4 restores, actual/attempted integration
counts, graph-build state/clock effects, external copy counts, raw/EFC packets,
private context and all original/new masks. It separates actual scaled gradient
from direct-M*qacc versus accumulated-Ma and force recovery, retaining original
velocity/contact/gravity terms. All scope/repeat raw-field differences are
reported. Partial query bytes and unknown setup work remain explicit on failure.
The historical sources lack separate saved policy161 vectors: the worker
verifies regenerated observations across full restores and input102/PD/reward
contracts; it does not invent direct historical observation-byte comparisons.

The graph/worker is still NOT GPU-qualified. Finish/review/aggregate all54 and
final videos first. Then run A/B serially under the frozen registration; first
queries count as qualification inside the same budget. Preserve any failure,
change source/version/registration explicitly if needed, and independently
review actual saved data before inferring a root cause. No physics/gate/
tolerance/integrator/backend/installed-source or driver/CUDA change is admitted
by these CPU checks. All original M0–M5/A1/A2, full-task, multi-reference/seed,
total-cost/performance, real-ego asset/gravity/dynamics/RGB and hardware
observation/control/latency requirements remain open.
