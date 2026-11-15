# Test action-table amplitude after full-orientation scoring fails to stabilize control

Previous turn is PROGRESS: v0.83 full-SO3 implementation and211 tests passed;
offline138 sources/7866 endpoints1006/1006,91 changed rankings; eight actual
cases add46168 integrations, all fail at0.30–0.36s. Matched matrix133/133,
saved contracts3043/3043, actual EFC56/56, selected forecasts131/132 (FULL_SO3
67/67, SHAFT force/torque failure retained). Coverage660/660: only69/132 visited
sources have any of57 endpoints within original10mm COM/10deg full angle plus
geometry gates. Best-safe versus zero-offset angle improvement median per
case0.104–0.300deg. This is finite-table evidence, not global infeasibility.
New actual videos inspected; Goal ACTIVE/incomplete, no neural/Actor update.

1. Test this restricted action-table hypothesis, not another orientation/CCD
   sweep. Predeclare amplitude multipliers1 and4 applied to the SAME original
   368x57x20 raw offset array (directions/order/seed8101 retained). Original
   axis offsets+/-0.08 become+/-0.32 raw; random offsets are exactly4x original,
   not new samples. Zero-offset center is identical. Do not introduce another
   implementation copy; default multiplier1 preserves legacy behavior.
2. Preserve raw action bound+/-1 and original .5 scale/.5 EMA, encoder bias,
   actuator limits and geometry/finite/capacity-first fallback. Scaling raw
   offsets is not an equal-size change in executed PD targets: record raw,
   EMA/intended/clamped PD differences and clipping explicitly. Validate the
   entire raw table plus fixed center before any execution. No object force,
   reset, gravity or wrist assistance; original initial pose/reference remains.
3. Fix FULL_SO3 at original5deg angular scale, original COM/hand/residual/support
   weights, CCD50/1e-6, constraint numerical policy,57 candidates and20ms horizon.
   Run multiplier1/4 x STATE_ONLY/STATE_PLUS_SUPPORT x dt5/2.5ms = eight fresh
   cases with identical source code and config except multiplier. No additional
   gain/horizon/initialization/CCD or amplitude sweep after seeing results.
4. Compare actual full7.36s/COM/full-orientation/geometry/joint/nonhand-force/
   no-aid gates, first native terminal and full survival cost; per-decision57
   is not equal total integration when duration differs. Independently verify
   raw/EMA/PD/reference/source/model/layout/import/clock/isolation, selected
   state/force/torque forecast and actual EFC/momentum at original gates. Retain
   every failure and source receipt. Stronger motion with inaccurate contact
   prediction is diagnostic/nonexpert, not gradient-qualified supervision.
5. For all recorded57 endpoints, repeat original10mm/10deg joint endpoint
   coverage audit, finite geometry, clipping and action effect ranges. Interpret
   coverage differences on diverged visited states as conditional observations,
   not the same-source causal response or whole-task/global feasibility proof.
   Preserve original RMS/duration gates and measured uncertainty. Save actual
   videos. No long blind Actor training or reference/gate redefinition.
6. If coverage and task performance remain poor, use the result to choose a
   source-matched control-domain or properly clocked longer-horizon investigation
   before learning. A longer native predictor must advance future reference,
   EMA and control at50Hz; extending one held-PD20ms graph without those states
   would not match actual execution. Do not silently substitute that shortcut.

Original learned SuperMimic+CHORD residual/support fair-total-budget A1/A2,
world-model contact/gradient verification, cube/pen/save-resume, multi-trajectory
adaptation cost, true-ego input/frame/assets/dynamics and hardware contracts
remain required. Diagnostic negative results do not complete them. No active
training/checkpoint wait, no new pause request. Full Goal ACTIVE/incomplete.
