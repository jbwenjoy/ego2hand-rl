# Clarifying physical-audit censoring in the first36 cases

The first36 cases'0/36 complete-task result and all original gates/masks remain
unchanged. The added read-only taxonomy classifies all6110 reviewed queries;
it does not run physics or revise any outcome. Report:
`runs/pen_screened_failure_taxonomy_36_v1/report.json`, SHA256
`e7c741867036e69afe467fdda341766f0afefed603f9a8e437b54b32f1b57de1`.

All12258 consistency checks pass: two per query,36 BASE stopping checks and
complete case/query coverage. Every query remains in `all_queries.csv`.
These checks regroup the already independently raw-decoded V3 results and
re-evaluate the original scalar threshold; they are not another raw decoder.

- 6069 queries are physically qualified;41 are invalid:19 BASE and22 ALT.
- All41 fail only EFC_linear_impulse. No other geometry, actuator, label, online
  or physical-audit condition fails in this checkpoint's queried packets.
- All41 invalid queries have terminated=false and truncated=false. All19
  physical case stops occur on their final BASE query, before a task termination
  flag. The original native-screening protocol therefore censors their future.
- Invalid maximum linear impulse residuals have min5.853724569e-7,
  median1.055800612e-6,p95 2.487888014e-6,max2.866139584e-6Ns.
  The unchanged original threshold is5e-7Ns. Their magnitude alone neither
  establishes a solver root cause nor authorizes a different tolerance.

Thus these19 case stops cannot be presented as object drops at the stopping
time, or as evidence of what their unexecuted future would do. The17 task
failures remain actual task failures, and none of the36 completed the registered
full7.36s qualified task. No eventual success is inferred for censored cases.
The frozen videos correctly freeze stopped panels, but that visual stopping
must be interpreted using the displayed audit/task reason.

This strengthens the priority of the already registered same-solve-exit
diagnostic. It still cannot distinguish actual scaled solver gradient from
accumulated-Ma drift, force recovery, velocity integration or recording epochs
without the planned native measurements. The original676-query partition and
all its failures remain unchanged; no new failing-tick private data exists.

Continue the same full54 producer and final independent review, then the frozen
bounded replay under its terminal prerequisites. No installed-source, physics,
integrator/tolerance/gate, driver/CUDA, model/normalizer or optimizer changes.
Research plan v1.48 and every original M0–M5/A1/A2/ego/hardware requirement stay
in force; this clarification does not complete the whole research goal.
