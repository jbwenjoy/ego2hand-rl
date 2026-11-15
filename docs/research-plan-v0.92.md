# Execute the declared horizon matrix as unqualified controller diagnostics

v0.91 implementation and matched-source pilot make PROGRESS.238 tests pass.
8/8 exact first PD/isolation/model contracts and111/111 scheduled independent
source/PD/no-aid/capacity contracts pass. H1/H5 first20ms comparisons441/456
pass;15 force/torque failures remain despite state/control/time agreement.
Full100ms graph/independent response fidelity47/111, EFC108/111. Full-horizon
state drift also fails gates. Do not call this dynamics-qualified or train on it.

1. Preserve pilot bytes/config/source receipts and all failures. Continue the
   predeclared H1/H5 x state/support x5/2.5ms actual native matrix as diagnostic
   execution only. No task/gradient/benefit qualification follows even if
   survival improves; no gate or initial pose/reference/clock is changed.
2. Add explicitly configured actual post_qacc_warmstart capture to NEW v2
   matrix configs, not to frozen pilot configs. This is boundary instrumentation,
   not a dynamics change or performance optimization. Independent review must
   compare source memory to actual previous post_qacc_warmstart and actual
   state/clock/PD, never substitute post_qacc. Last source memory is recorded
   as well. Legacy default recording behavior remains unchanged.
3. Review future raw/EMA/reference/bias/limits and per-control state/support
   score independently with NumPy; check original fallback/safety and sum all
   forecast/actual integrations. Compare ONLY forecast first20ms to actual
   executed control. Later forecast controls are hypothetical and cannot be
   compared to subsequently replanned actual controls as if they were executed.
4. Independently decode every actual episode EFC and original full task gates.
   Keep control/clock/isolation/score checks distinct from force fidelity and
   task success. Old20ms-only readers explicitly reject H5; no wrong endpoint
   reference or warmstart substitution. Render actual traces if useful.
5. Fixed238 tests and successful source/clock checks are implementation evidence,
   not a physical pass. Preserve role CONTROLLER_DIAGNOSTIC and never fit these
   episodes. Full Goal ACTIVE/incomplete, no new MLP/Actor or real ego claim.
