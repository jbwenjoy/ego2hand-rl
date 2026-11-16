# Native controller/reward-chain audit V2 — serialization repair only

V1 completed all4860 saved-row reconstructions but its final JSON serialization
failed because a summary count was numpy.int64. Preserve original source/log/
failure and all saved arrays/context. V2 changes only that count's serialization
to Python int,with fresh registration/output paths. All math,inputs,channels,
tolerances and physical masks stay byte-equivalent. Compare both versions'
saved numeric arrays and source contexts before accepting V2. This is not a
backend/physics revision or new-data collection;both attempts have0 native,
optimizer and normalization steps. Original registrations/source bytes stay.

Read all already-reviewed new-backend TRAIN/validation sources and branches;
preserve all source/branch/row/group exclusions and both original roles. Prior
six-model forward/JVP results and one original BASE formula preflight are viewed
development evidence,not unseen holdout. Freeze this full decomposition before
aggregate native scalar contrasts;no optimization or model selection is performed.

Verify exact source metadata/hidden backend/receipts. Reconstruct original RAW
clamp[-1,1],scale.5,residual clamp[-.6,.6],EMA alpha.5,reference+EMA-encoder-bias,
then actual joint/actuator target-limit intersection. Masks must be disabled.
Resolve actual transmission control order from saved state;single source trajectory
starts0 and phase/index must agree. Every observed branch actualPD agrees within
original2e-7rad;all8 actual PD samples must be byte equal to recorded actualPD.
At unclipped interior points dPD/dRAW=.25;the original signed PD amplitudes
.002/.005 correspondRAW amplitudes.008/.020. Finite scalar responses divide by
2*RAW-amplitude=8*PD-epsilon,not2*PD-epsilon. Excluded clipped sources remain
analytic source diagnostics only,not invented measured responses.

Use maintained original tracking/objective math to reconstruct saved tracking,
geometry loss/failure,source-frame support target/measured wrench/bonus/eligibility
and terminal flags. Recorded post-physics/reference phase is used,not next phase.
Comparison tolerances:tracking5e-6,geometry2e-6,bonus2e-6 dimensionless;support
loss1e-8+1e-10*abs(original),target/measured-source1e-10. These only compare CPU
formula reconstruction with saved floating outputs;original physical gates and
5e-7Ns impulse masks are unchanged. A BASE formula preflight passed without
adjusting these tolerances. No native forward or integration is allowed.

Report weighted COM/SO3/hand/action components,recorded tracking,geometry penalty,
centered unweighted bonus,and both fixed original residual0/support.2 objectives.
Full reward is a reconstruction of recorded components;no separate raw reward
tensor was saved. Do not present it as an independently logged raw reward.
Keep all complete/empty/physically invalid groups. Summaries use original physical
masks only. Scalar amplitude screens are descriptive:both amplitudes qualified,
relative error<=.25,cosine>=.9 and contrast exceeds two-repeat envelope at both
amplitudes. No signal or missing group has no invented ratio/direction. Use unit1
dimensionless reward values;no normalizer fit or TRAIN pair selection.

Current WM90 predicts endpoint state/engine memory and mean source-BODY wrench/
Fn6. It does not predict max penetration/limit excursion/nonhand load across8
substeps or all geometry/global-error flags. These path terms cannot be replaced
with endpoint values or dropped while calling a proxy the full objective. Report
geometry's share as abs(geometry change)/(abs(tracking change)+abs(geometry change)
+ .2*abs(bonus change)),not a cancellation-prone signed fraction. Fn6 is not a
direct argument of the current centered mean-wrench score. A differentiable
endpoint/mean-wrench observable proxy may be tested,but this audit does not admit
its reward gradient,full Actor,task,ego or speedup. Full original M0–M5 and all
multi-reference/seed/ego/hardware requirements remain active.
