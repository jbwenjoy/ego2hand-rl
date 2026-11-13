# Isolate stopping tolerances, then full-case numerical-quality validation

Previous turn progressed ined8a759: fixed action envelopes cover18 cases but
all actual tasks fail0.16–0.30s; EFC124/126 retains two impulse failures.
Local TIGHT changes improve fine384/384 but coarse324/384, so no complete
repair is accepted.166 tests pass. Full Goal active; no neural training yet.

At the SAME two offending saved states use128 worlds/three one-step repeats
and all original input/model/force/impulse gates. Six fixed variants:DEFAULT;
CAPS50(main/LS limits50, default tolerances); TOL_ONLY(1e-10); LS_ONLY(1e-6);
BOTH(these two); BOTH_STRICT(tol1e-12,LS1e-8). All nondefault variants hold both
iteration caps50. No physical parameter, timestep/integrator/CCD edit. Keep
all outcomes and source revisions; no additional parameter search this phase.

Choose the least-tight preregistered variant passing BOTH source states in
ALL384 worlds each, in order CAPS50,TOL_ONLY,LS_ONLY,BOTH,BOTH_STRICT. This
is numerical calibration on known failures, not an independent success test.
If none passes, keep the failure and do not silently relax gates.

Run fresh actual original-clock episodes at BOTH dt for DEFAULT and the
selected variant, fixed profiles BASELINE,D4_PLUS,D2_PLUS,D2_MINUS. Use the
existing bounded actions/EMA/selected pose and first termination. If none
passes calibration, run DEFAULT and the preregistered BOTH_STRICT diagnostically,
with no acceptance claim. Preserve all failed trajectories. Record actual
solver iteration counts and requested/host/GPU settings. Compare physical
tables, source/control phases, all task gates and actual EFC/momentum receipts;
timing is descriptive, not training speed or control benefit.

Shared implementation accepts only the four explicit numerical keys, applied
through task configuration BEFORE environment/graph construction. Reject
gravity/mass/geometry/integrator changes. Preserve prior evaluator/probe bytes.
No installed engine/driver/CUDA changes, post-start resets, object aid, Actor
fit or conversion of CHECK diagnostics to expert labels. A local passing
probe cannot qualify a full episode or100ms deterministic repeatability.

All original controlled training/save-resume, matched residual/support and
WM action gradients, ego/multi-trajectory/total cost/hardware scope remains.
Future neural training stops at first healthy checkpoint for restore/ETA and
user-requested Goal pause.
