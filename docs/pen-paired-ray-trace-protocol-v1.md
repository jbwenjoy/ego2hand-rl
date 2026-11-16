# Prospective paired-phase ray trace v1

This is a new bounded diagnostic after the closed1,920-integration termination
comparator. The following1,920-integration cap is prospective, not a registered
or consumed budget. Factory/CPU qualification does not authorize an unimplemented
worker to claim completion. Keep all original M0–M5/true A1/A2 requirements.

## Why another observation is needed

The exact pinned _solver_iteration calls line search, constraint update,
gradient/search update, then the stopping kernel. End-of-solve alpha and
improvement describe the preceding line search, while saved grad/search and
decrement describe the later update. Their dot product is not automatically
the derivative of that preceding ray. Also ls_exhausted means
abs(alpha)<noise_floor, not proof the LS_ITERATIONS loop reached its cap.

The existing quadratic pyramidal/equality implementation already computes
shifted cost differences; friction evaluation also has absolute-cost branches.
Do not presume naive total-cost subtraction, missing gradient rebuilding,
wrong search direction, or an FP32 root cause from the endpoints alone.

## Minimum aligned payload

Index every payload by process,algorithm,source,repeat,scope,control tick and
outer solver iteration0–49. Record both ne/nf/nefc and valid-prefix counts;
never interpret zero-filled unused buffer tails as observations.

1. At the actual iterative line-search entry, copy raw grad/grad_scale, grad_dot,
   decrement,search,mv,Ma,smooth,qacc. Copy each active row's Jaref,computed jv,D
   and frictionloss after fused jv is available. Use the actual computed ray
   inputs; do not regenerate geometry/J/M or borrow final env-forward caches.
2. Copy actual GPU row/Gauss reductions,p0,gtol,noise_floor,snorm,Newton alpha,
   first evaluated delta/derivative/Hessian and initial acceptance predicate.
3. For each actual bracketing iteration0–49, copy entry/trial/exit alphas and
   points,swap flags,ls_done,improved,best alpha/delta,accepted alpha/improvement.
   Preserve all predicates so no-swap stopping,derivative stopping and reaching
   the loop cap can be distinguished. Initial direct acceptance has zero
   bracketing rows. A noise-floor flag alone cannot identify a stopping branch.
4. After the same outer iteration's gradient/search update, copy grad/scale,
   search,grad_dot,decrement,state/quad-change counts,search_unchanged,qacc/Ma
   and solver_niter. Pair it to entry index using the measured native iteration
   counter, not host-side guessing. Post-update copying remains unimplemented.
5. Retain the existing complete source/end state,raw contact/EFC packet and
   solve-exit copies,all original/new masks and task flags. Source restores and
   actual RAW/PD stay exact. No trained module is loaded or fitted.

The factory generator in shared src emits only diagnostic writes and scalar
locals. It checks the complete canonical generated AST and verifies the
original physical AST is exactly recovered when diagnostics are removed.
External helper aliases must be the exact pinned original objects. All buffer
owners remain outside the native/Python owned runtime. Clear valid flags at
each solve without advancing physical time. Overflow increments a diagnostic
error flag and does not skip/alter the original physics; reject the trace if
overflow/missing indices occur. Allocate full njmax4096 unless an explicitly
proven bound replaces it; the observed context is W1,nv26,nv_pad28,Newton2,
pyramidal0,dense,implicitfast3,outer50/line-search50.

AST equality and CPU parity are not GPU neutrality. New writes can change
compilation/scheduling. Use a separately identified observation implementation,
retain both graph/context owners,attest temporary bindings and cleanup, and
check exact source/clock/counters around every graph capture. The installed
engine sources and driver/CUDA stay unchanged. Preserve failures and unknown
partial work; no restart of a partial process.

## Scope pairing and budget

Use all ten historical conditions from pen_solver_tick_preparation_v1,including
unresolved failures,rescued shared-source SHAM conditions,old passing controls
and the condition with the newly observed candidate failure. All are inspected
diagnostic inputs,not blind test/fitting data. Preserve duplicate source/actions.

Compare INSTALLED_ORIGINAL_OR and DROP_IMPROVEMENT_ONLY without changing either
rule,50 cap,tolerances,physical parameters or original physical/task gates.
Use two observation scopes:

- SOLVE_EXIT_ONLY: the current qualified solve-exit collector,providing endpoint
  copies for every baseline query.
- SOLVE_EXIT_PLUS_RAY_TRACE: the same endpoint collector plus aligned ray and
  post-update payloads.

This isolates the additional trace against the existing observer. It does not
establish neutrality against an unobserved simulation; earlier packet-only
scope variability remains part of the evidence. Each fresh process has10
sources x3 numerical repeats x2 algorithms x2 scopes =120 queries/960 native
integrations. Two processes total240/1,920. Reverse algorithm blocks in B and
counterbalance scope order. Register exact schedule,inputs,implementations and
independent reader before execution. Zero optimizer/normalizer work; no extra
native setup allowance. First CUDA qualification consumes the same budget.

## Independent replay and prerequisites still pending

pen_ray_reference_v1 independently evaluates the fixed recorded pyramidal ray
objective in FP64,including equality,unilateral active-set changes and friction.
It compares actual recorded trial alphas and points; it does not prescribe a
replacement step or claim higher-precision physical truth. It uses recorded
FP32 jv/mv/Jaref/Ma,so it cannot diagnose unrecorded J/M computation by itself.
Unsupported inputs (for example a friction row with nonpositive D) must stay
explicit failures,not be dropped or silently coerced.

Before live work,implement and test the coupled observer/post-update copier,
source/graph ownership and alias guards,explicit observation identity,worker
and independent saved reader. Reader must recompute actual trial objectives,
the recorded GPU decisions/valid prefixes,phase alignment and original impulse
arrays; keep scope changes,regressions,nonrescues and all costs. A full failure
classification may require additional data; no unique cause is promised here.
No native trace registration/execution is established by this protocol.
