# Long-control final evaluation v1

Register before either final checkpoint is inspected. Finish both fixed2000
training arms without checkpoint selection or budget changes. Use the shared
qualification's common actor/norm initialization plus RESIDUAL2000 and
SUPPORT2000, in that order, three fresh W1 executions each. These are numerical
repeats of one trained seed/reference, not independent training seeds.

Construct a fresh SupportPenRuntime with the exact training objective and
adapter config. Switch fixed-frame features to LIVE before any observation or
step used for evaluation. GPU1 CACHED has not been qualified. Copy only the
actor and its embedded frozen normalizer. Never restore W32 native Model/Data,
critic, Adam, RNG, counters or historical probe physical state into W1.

Check full-pose161, raw20 clamp/scale/EMA/PD, joint order, objective arm/weight,
calibrated finite weights, source receipts, engine bytes, reference and host
topology. Require every current W1 Model array (including shape/dtype/bytes)
and scalar to equal the frozen historical W1 plant witness. That witness's
historical actor pin may differ explicitly: only Model constants are compared.
Do not broadcast W32 solver constants into W1 or load any old witness state.

Qualification uses the first optimized RESIDUAL checkpoint and checks this
weights-only admission without advancing physics or optimizing. Report the
observed native step count, and retain failures. It is preparation for final
evaluation, not evidence of policy performance or CACHED1 qualification.

Use existing controlled_evaluation_v4 gates unchanged: full7.36s, COM RMS5mm/
max10mm, full orientation RMS5deg/max10deg, object penetration2mm, self0.2mm,
joint excursion0.001rad, nonhand normal0.1N, finite/capacity/no-applied-force.
Fixed wrist, original2.5ms physics/20ms control, one constructor reset, no
post-start reset; stop each case at the first terminal/truncation. Keep all
failed outcomes and reasons. Native raw contacts, forces, actions, used phases,
pose/velocity, control windows and model assets are saved for each execution.

Report all nine full-trace outcomes and durations. Also compare trained arms'
COM/full-angle errors over the same shortest observed trained-arm horizon;
these prefix errors are separate from full-task acceptance. Record actual
native integrations, zero optimizer steps, and evaluation/record/render wall
time. Rendering replays saved2.5ms poses only, no simulation or interpolation.
Use repeat0 for the three-panel20x-slow video regardless of its outcome. Keep
terminal stills labeled. Do not overwrite the old64 video or claim ego results.

Source and input pins are verified before/after. Exceptions produce a failure
record with the native work observed so far, without replacing samples or
silently restarting. New evaluation files are unimported by the ongoing trainer;
its captured source/config/protocol must remain byte-identical.
