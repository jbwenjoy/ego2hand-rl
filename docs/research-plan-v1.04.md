# Research plan v1.04 — first geometry/support pilot and score audit

Previous goal turn made concrete progress: equal-budget actual geometry/support
training, fresh W1 mean executions and an inspected three-panel video. This
turn adds complete-state/actual-continuation review214/214, independent original-
clock pose/raw-EFC/wrench review114/114, offline reward-signal audit10/10 and329
unit tests. Keep the complete v1.03 objective, gates and all failed evidence.

1. Archive exact source/report/checkpoint pins for both qualifier versions,
   training, continuation diagnostics and both preview executions. Preserve the
   initial qualifier's uninformative optimizer check, replacing its evidentiary
   role with the final directly checked v2. Current training3 is actual
   geometry/support pilot training, not the old tracking-only diagnostic.
2. Treat v1 as a failed full task, not a null or positive statistical support
   result. Fresh residual/support both terminate.24s; first full-angle breach
   .1225s and COM breach.145s. Mean-execution geometry remains legal, but sampled
   training includes violations. Always compare the common prefix and full
   original horizon; do not equate short geometry compliance with success.
3. Before extending training, preregister OFFLINE reward variants on the same
   saved rows: current absolute score, contact-load eligibility, and zero-hand-
   wrench-centered score exp(-L/2)-exp(-L0/2). Quantify no/weak-contact behavior,
   force/torque sensitivity, nominal-target/physics feasibility and the effect
   of a contact threshold. Gating creates a discontinuity and can incentivize
   squeezing; retain object/self/joint/nonhand maxima. Centering is a changed
   objective and may alter episode-length incentives; do not assume it preserves
   optimal policies. No claim of improved policy before new actual execution.
4. Select and freeze a new reward contract only after those checks. Give any
   changed objective a new explicit config/schema tag and reject old implicit
   loads; use common-source initialization, normalization and equal TOTAL
   native/data/update budgets. Include calibration/qualification/diagnostics/
   failed runs and evaluation. Do not continue the v1 pilot and then pretend its
   preregistered budget or objective is unchanged. Do not keep merely adding
   epochs to the current weakly distinguishing support score.
5. Diagnose controllability/contact-feasible reference under original compound
   geometry/mass/inertia/friction/clock. Measured event ordering is descriptive;
   use controlled matched interventions before blaming contact loss, EMA or
   target noise. Hand wrench targets are nominal prior rewards, not body aids.
   Keep fullSO3 tracking, moving-COM/source-axis and USED-interval contracts.
6. Revisit physical WM forward/action-gradient qualification on contact-rich,
   geometry-legal shared data. Existing failed WM/noise/finite-difference gates
   are not superseded by model-free support reward implementation. Preserve
   W32 source4/5/6 and repeat/noise issues as separate numerical requirements.
7. After objective semantics settle, profile integrated native/contact/cache/
   collector/update guards with matched data and budgets. Cold validation and
   temporary instrumentation times are not an end-to-end speedup result. Keep
   pending/finite/reset/stream ownership until equivalence evidence justifies
   changes; do not patch packages, drivers or CUDA.
8. Full remaining objective unchanged: original learned cube/pen no-aid task and
   unified save/resume/eval; common multi-trajectory/multi-seed residual/support
   A1/A2; contact/dynamics uncertainty and physical WM gradients; real ego frame/
   geometry/physics assets; hardware subversion,observation,zero,gain,latency
   contracts before deployment. Controlled perfect poses are not ego data.

Goal ACTIVE/incomplete. The historical first-checkpoint pause was already served
and explicitly resumed; no new pause is requested. Continue locally from the
archived negative result and the next objective-validation step.
