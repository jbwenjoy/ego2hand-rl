# Test feedback support without changing native control budget or physics

Previous turn is PROGRESS: four native finite-action episodes and independent
review, descriptive tracking diagnostics, decoded actual videos and194 tests
are complete. All tasks fail; selected force replay68/70 and actual EFC27/28
also retain failures. Full Goal remains ACTIVE/incomplete.

The nominal reference feedforward target does not correct actual state errors.
It has no consistent survival/tracking benefit. Final errors are mainly
transverse; positive hand load often persists. These observations motivate a
single-factor support-target test, not a proven failure cause or weight sweep.

1. Keep original initial pose, reference clock, fixed model, no-aid/controller
   mappings, native20ms horizon,57-candidate table/seed8101, score scales/weights
   and all geometry/task/force/replay gates. No native live reset or applied
   corrective force; the following force is only an objective target.
2. Add STATE_PLUS_FEEDBACK_SUPPORT alongside original STATE_ONLY and
   STATE_PLUS_SUPPORT. WORLD target about object COM:
   F_target = F_nominal + m*[100*(p_ref[k]-p_source) +20*(v_ref[k]-v_source)].
   Gains correspond to predeclared wn10/s,zeta1. Retain nominal torque,
   original derivative-valid masks and0.1 support weight. No clipping,
   fabricated measured forces or state-dependent gain tuning. Match source
   time k, not endpoint k+1. Rotation/angular feedback is outside this one
   ablation. At zero source error target exactly equals nominal feedforward.
3. Implement target arithmetic once in shared src with device-resident source
   views and reference tensors. Save exact source errors, correction and target
   at every decision. Test zero-error equivalence, restoring direction/units,
   mass scaling, rotational covariance and nominal torque preservation before
   actual collection. Preserve old frozen runner/reviewer bytes and records.
4. Run all six cases until first original native terminal or full7.36s clock,
   with equal57 candidates/decision. Report full native ticks, survival and
   common-prefix metrics; unequal survival still means unequal total cost.
   This is development evidence on the same controlled clip, not a held-out
   test, learned-policy A1/A2 or true ego generalization.
5. Independently reconstruct every target/cost/action from saved source state,
   compare selected forecast with actual physics and retain every EFC/task/
   fidelity failure at original gates. No failed prefix becomes expert data.
   If feedback does not help, do not run a gain sweep; inspect sampled action
   authority, horizon/shortsightedness and actual velocity-control requirements
   before choosing a distinct next intervention.

This test probes contact-support objective design while learned gradients
remain unqualified. Full original input/frame/asset/dynamics contracts,
learned control/save-resume, cube/pen evaluation, fair multi-trajectory total
cost, real ego evidence and hardware contracts remain required. No new pause
request or training/checkpoint wait exists; no automatic goal pause/complete.
