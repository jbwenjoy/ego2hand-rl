# Closed paired ray trace and fixed-input friction arithmetic

This experiment adds aligned observations,not a trained policy or a solver
repair. The prior termination comparator and all54 full-task outcomes remain
immutable. Main M0–M5/true A1/A2,ego and hardware requirements remain open.

## Complete execution and review

Shared source now implements paired graph ownership,external buffers,
same-iteration post-update copies,exclusive reversible Python callbacks,
source/alias guards and independent recorded-decision replay. The actual runtime
class and owned numerical inventory remain unchanged;historical states retain
exact native/Python/RNG values with the already explicit backend metadata import.

Full617 CPU tests pass in11.942s,including actual post-copy kernels,cleanup,
pointer overlap rejection and replay of recorded CPU primitive traces. The
fixed-friction arithmetic probe has two additional focused CPU tests. These
test stages remain separate;no full619-test suite is claimed.

Registration runs/pen_paired_ray_trace_registration_v1/registration.json has
SHA256 5a943bf58e3f441908f7a3fc0ecc44161560d8dc94204b8bfa2b314fde6e4cb2;
20 exact historical-source import preflights pass. Independent reader
registration SHA256 is
ea8ca752091f84dc6b1d89a85b767397d499aed3ac5570c753db6af4248dd30e.

Both fresh processes complete120 queries/960 actual=attempted integrations,
7/7 producer checks each,zero setup advances,optimizer/normalizer updates.
Each process builds four owned pairs of graphs (eight graphs) without changing
native/Python/RNG/clock/counters during capture. Both saved readers pass
4,399/4,399 root checks each and independently recompute141,120 force/impulse
array elements each. Nested replay checks total183,660 and all pass.

All240 queries retain the same ten inspected conditions,exact RAW/PD,
three numerical repeats,two algorithms and two observation scopes. Both scopes
save endpoints:SOLVE_EXIT_ONLY versus SOLVE_EXIT_PLUS_RAY_TRACE. B reverses
algorithm blocks. This isolates additional tracing against the earlier
endpoint observer,not against unobserved physics. No independent training seed,
blind test,population incidence or ego generalization is established.

There are6,958 actual coupled rays (A3,717/B3,241),with59,618 reference point
records. Repeated trial alphas stay in the records;the reader caches their
fixed-input FP64 evaluation within each ray. These are not59,618 independent
conditions. Every actual bracket swap,acceptance and stopping decision is
replayed from the recorded FP32 points. FP64 disagreement is a measured
outcome,not a newly relaxed physical or verification gate.

## Physical masks and observation effects

| Condition | Original OR invalid/12 | Drop-improvement invalid/12 |
|---|---:|---:|
| SUPPORT_P058 |12|12|
| SUPPORT_P057 |0|0|
| UNMODIFIED_P027_BASE |0|0|
| UNMODIFIED_P026_BASE |0|0|
| SHAM_P064_ALT |6|0|
| SHAM_P064_BASE |6|0|
| SHAM_P110_BASE |1|0|
| SHAM_P109_BASE |0|0|
| RANDOM_P067_BASE |12|12|
| RANDOM_P066_BASE |0|0|
| Total |37/120|24/120|

These are this observation experiment's results,not a replacement for the
previous47/120 versus25/120 termination-comparator results. SHAM_P064 duplicates
share a source/action. There are24/170 cap50 ticks for original/candidate over
960 endpoint-observed ticks each. No query terminates/truncates within its
single20ms control;this does not measure a complete task.

Only91/120 matched observation-scope pairs are byte-identical;119/120 have
equal impulse masks. The unmatched mask is retained. Additional tracing is
not proven neutral. The original5e-7Ns impulse gate and all other physical/
task thresholds,parameters,integrator,actions and solver limits stay unchanged.

## Same-phase cost rejection is now measured

All30 failed-query peak rays in the traced scope (18 original,12 candidate)
have a GPU initial trial cost delta>=0,a negative FP64 delta at the same
recorded inputs,and a recorded trial derivative already within gtol. Thus
the GPU's nonnegative cost blocks that initial acceptance predicate in these
observations. This statement no longer mixes the previous ray with the later
updated search direction. Both the two-step original stopping and cap50
candidate outcomes remain in the data. All these peak line searches use two
actual bracket iterations;ls_exhausted is still a noise-floor flag,not that count.

Example:RANDOM_P067_BASE,process A,original algorithm,repeat0,peak tick0,
outer iteration3. The actual pre-ray G dot search is-7.436082963e-7;
GPU/FP64 ray slopes are-7.434282452e-7/-7.435105818e-7. At alpha0.9997580051,
the GPU reports cost delta+1.692504156e-6 while FP64 gives-3.717064480e-7.
The recorded trial derivative is-5.820766091e-11,below gtol9.999999975e-7.
There are20 friction rows. Illustrative FP32 subtraction of row costs around
4–11 can lose micro-scale changes;the actual paired trace captures the full
GPU reductions and decisions rather than inferring them from endpoints.

Across all6,958 rays,4,388 initial cost signs disagree with fixed-input FP64;
many are tiny changes in otherwise valid queries. Do not treat that number as
a failure rate or assume every disagreement matters to a task.

## One fixed-input CUDA arithmetic probe

After both native processes and reviews close,the standalone probe evaluates
all139,160 friction rows at every coupled ray's initial Newton alpha. It has
a predeclared2,000,000-row cap,one CUDA primitive launch and zero integration,
forward,policy or optimizer calls. Inputs stay byte/numerically unchanged.
No helper is rebound into the simulator.

The experimental formula uses shifted quadratic differences or +/-f*advance
when start/end stay in the same friction regime. It covers137,144 rows;
2,016 cross-regime rows explicitly retain the original computation. Original
versus experimental gradient/Hessian outputs are byte-identical for all rows;
fallback outputs are also byte-identical. All8 analysis checks pass.

| Absolute row cost-delta error versus FP64 | Original | Same-regime + fallback |
|---|---:|---:|
| Median |6.326624792e-9|3.722651649e-15|
| p95 |2.387763331e-7|1.329270416e-8|
| Maximum |2.803589763e-6|1.447312229e-6|

This localizes a concrete arithmetic weakness in absolute friction-cost
subtraction. It does not prove that substituting the formula in a live solver
will repair all failures. Standalone primitive compilation/reduction can differ
from inlined native code. Replacing only its friction difference in the observed
aggregate illustratively turns all30 failed peak cost predicates negative;
that substitution is not a re-executed line search or a physical transition.
Cross-regime fallbacks and remaining maximum errors stay explicit.

All query/ray/scope CSVs,fixed input/output NPZs and source receipts are in
runs/pen_paired_ray_friction_analysis_v1/. The V2 figure at
runs/pen_paired_ray_friction_figure_v2/cost_signs_and_friction_error.png was
visually inspected. V1 remains preserved;V2 uses hollow circles/crosses so
coincident observations of both algorithms remain visible.

Native-process walls total848.1578s (A422.8498/B425.3080),saved-reader walls
119.1902s. Arithmetic analysis wall8.1294s includes its0.4660s CUDA arithmetic
pipeline (upload,JIT,launch,download/readback). Timings overlap some CPU analysis,
exclude other preparation/tests/plots and are not complete adaptation cost or
isolated solver/training speed benchmarks. Installed sources/driver/CUDA were
not changed.

## Next: separately identified numerical-cost candidate

Implement a new experimental backend that changes only same-regime friction
cost-difference evaluation in the single-alpha and three-alpha pyramidal
helpers. Preserve derivatives/Hessians,cross-regime fallback,original OR stopping,
cap50,parameters and physical gates. Use an isolated pinned factory and explicit
binding/metadata identity,not an installed edit or accidental old factory-cache
reuse. Do not promote the drop-improvement candidate or train under this probe.

After CPU/GPU primitive and source/restore/graph checks,register a bounded
original-versus-stable-cost native comparison on all ten sources,including
healthy,rescued and regressed cases. Measure both packet-only and endpoint
observation scopes,all failures/iteration costs and independent saved physics.
A new240-query/1,920-integration cap is prospective,not yet registered or run.
Require physical-backend requalification,including intended training world
counts,before recollection or Actor training. Full-task/reference feasibility,
true A1/A2,independent seeds/references,GPU performance,ego asset/dynamics/
gravity/RGB and hardware observation/control contracts remain open.
