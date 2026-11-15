# One velocity-tracking ablation before a longer native planning horizon

Previous turn is PROGRESS: six native feedback-support episodes, independent
target/action/replay audit, all-candidate authority analysis and197 tests.
All tasks fail; feedback shows no consistent benefit, EFC41/42 and selected
force replay101/102 retain failures. No Actor update, expert or ego result.
Full Goal remains ACTIVE/incomplete.

The20ms cost has position and directed-shaft error but no object velocity.
Saved initial candidates expose a position/velocity tradeoff, while late
candidate envelopes are too small to correct accumulated errors. One declared
offline velocity rescore changes15/102 decisions (4/30 early). This is a
testable omission, not evidence of closed-loop benefit or a reason for a sweep.

1. Add ONE endpoint WORLD linear-velocity term to the current state score:
   sum(((v_end-v_ref[k+1])/0.25m/s)^2), weight1. Scale is the existing5mm
   COM scale divided by the existing20ms horizon, not tuned on survival.
   Preserve the reference derivative-valid mask; no extrapolated tail.
   Keep every original geometry, task, force/replay and no-aid gate. Full
   orientation remains separately reported and required by unified evaluation.
2. Run original STATE_ONLY and STATE_PLUS_SUPPORT controls plus their two
   velocity-augmented variants at both dt, using the SAME57 candidates/seed,
   pose, model, clock, horizon, EMA and support weight. Do not combine another
   feedback gain or alter actuator/material/gravity settings. Report full
   native work and common prefixes; unequal survival is unequal total compute.
3. Keep one maintained native planner and score implementation in src. Before
   editing source pinned by a completed experiment, preserve exact bytes and
   record explicit original-to-archive or Git-object identity. New executions
   should pin immutable source snapshots alongside observed active-source
   paths and code revision, verifying equality before/after collection. Do
   not silently reinterpret old absolute pins as current source or patch
   historical study copies. No system/installed-environment changes.
4. Independently reconstruct velocity targets, state/support costs and chosen
   controls from saved source/model/reference arrays. Check EVERY selected
   native20ms prediction against actual execution, including force/torque.
   Preserve EFC/task/replay failures even when the control contract passes.
   Failed prefixes remain nonexpert and never become imitation labels.
5. If the fixed velocity term does not yield a qualified task, do not sweep its
   weight. Inspect a longer native horizon with actual20ms reference/EMA/PD
   phase advancement and budget/fidelity accounting. A constant-PD60ms fork
   is not equivalent to three original controller steps. Any future longer
   forecast needs its own source/sequence/repeatability audit; current20ms
   checks do not pre-qualify it. Sampled authority remains a local diagnostic.

These are bounded development probes for a contact-correct controller, not
a replacement for learned SuperMimic/CHORD, cube/pen save-resume/evaluation,
fair shared multi-trajectory costs, valid world-model gradients, true ego
assets/dynamics/input contracts or hardware verification. No training is
running/waiting, no new pause request exists, and Goal remains ACTIVE.
