# Same-tick solver diagnostic preparation: CPU results

The existing full54 producer remains separate and unchanged. This checkpoint
does not capture private solver gradients or run new native physics. It adds
the arithmetic and input preparation needed for the next bounded diagnostic.
All prior v1.45/original M0–M5/A1/A2 requirements and failure masks remain.

`runs/pen_solver_tick_preparation_v1/report.json` passes103/103 on all5 retained
impulse failures from the fixed676-query cohort and all5 declared comparison
conditions. The configuration is `configs/pen_solver_tick_observation_v1.toml`,
and the execution/capture requirements are specified in
`docs/pen-solver-tick-observation-protocol-v1.md`.

The original raw/EFC decoder and its masks are recomputed unchanged. Every
failure is still an original impulse failure; all comparisons pass that audit.
Four comparisons use the previous control with a different source, while SHAM
P064 ALT/BASE share a source and nominal action. They must not be conflated as
five same-state contrasts. Source native4 hashes/trees, model, matrix mapping,
time step and solver options are inspected without stepping/forward dynamics.

All10 sources have26 DOFs and actual packed float32 M shape(1,71), despite
model.is_sparse=false. The
new pure CPU helper uses actual M layout and explicit row/column/address maps.
Its product agrees with a separately reconstructed dense matrix, which is
symmetric and has the expected free-body linear mass block. These are source
snapshot checks; they do not measure M or accumulated Ma at the failing tick.
Source cached-Ma differences cannot identify or exclude the failure mechanism.

The five-term same-solve-exit algebra distinguishes actual scaled gradient from
accumulated-Ma drift, force-recovery mismatch and smooth/gravity contributions.
The six new tests include an important counterexample: zero gradient and zero
stored-Ma force-balance residual can coexist with a nonzero physical residual
when Ma differs from direct M*qacc. Other tests cover actual grad_scale, separate
signed terms, packed/dense products, malformed data and overlapping terminal OR
predicates. Do not assign one termination cause from the final scalar values.

Focused6/6 and full568/568 tests pass; the full suite took11.054s. The test process
explicitly disables CUDA; its CPU-only Warp initialization warning does not
describe the separate active RTX5090 evaluation. CPU preparation took5.6565s,
with0 native integrations,0 optimizer steps and0 normalizer fits. These times
are not total adaptation or GPU performance measurements.

Preparation report SHA256:
`338a737a467be21e5d611cbc4496082841ae6b0afbb7ec93b2f742f7cf61331a`.

The proposed future observation has120 declared control queries across two
fresh processes, three numerical repeats, both observer scopes and all10
conditions: maximum960 native ticks,0 updates. It must reject execution until
both the full54 producer and final V3 review are closed successful workflows.
The process-local copy hook itself still needs implementation and GPU
qualification. Never modify the live producer or installed sources to add it.
Record exact same-solve-exit data before integration, plus original post-tick
packets; a later end-of-control forward cache is not a substitute. Record
observer effects, all changed/non-reproduced failures and complete costs.

No root cause, label fix, gradient/Actor admission, ego dynamics or acceleration
claim is made. Continue full54 closure and review, then implement/qualify the
bounded observation. Do not tune on the inspected first18 and silently call it
held-out validation; subsequent multi-step fitting needs explicit trajectory
ownership and renewed evaluation.
