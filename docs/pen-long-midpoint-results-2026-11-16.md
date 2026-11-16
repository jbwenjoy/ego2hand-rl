# RESIDUAL1000 midpoint diagnostic — 2026-11-16

The fixed RESIDUAL1000 checkpoint was registered before it existed; one fresh
LIVE W1 mean-policy episode has now executed. This is an intermediate single
episode, not the final long-training result, an equal-budget SUPPORT comparison,
an independent seed, or ego data. Finish the original two2000-round budgets
and registered nine-case final evaluation regardless of this diagnostic.

Observed outcome:93 controls,744 native integrations at2.5ms,1.86s until the
original object-position termination; no truncation. Full7.36s task fails.
COM RMS8.183mm/max50.710mm; full orientation RMS0.139489rad/max0.972214rad.
All four position/orientation error gates and full-duration gate fail.
Object penetration max1.9843mm, self penetration0, joint excursion0,
nonhand force0; geometry/no-applied-force/finite/capacity checks pass.
Do not describe1.86s as a qualified successful duration.

Actor/normalizer were copied into a fresh runtime only after complete404-array/
195-scalar W1 plant compatibility validation. No old native state,critic,Adam,
RNG or counters were imported. No post-start reset or object force assist.
Actual diagnostic optimizer steps0; execution+recording18.1618s,
saved-pose rendering13.8800s. Count744 integrations as separate diagnostic
cost, outside training and its4096-integration qualification cohort.

Report:`runs/pen_long_control_midpoint_diagnostic_v1/report.json`.
Video:`runs/pen_long_control_midpoint_diagnostic_v1/residual_1000_diagnostic.mp4`.
The two panels are different camera angles of the SAME episode,20x slow;
they are not two policies. Every displayed pose is from the saved2.5ms trace,
without interpolation. Terminal stills are labeled. Encoded video frames and
dimensions were decoded and checked; decoded middle/end images inspected.
Old64 and final-long artifact paths remain separate and immutable.

This new evidence shows that a longer intermediate model-free policy still
has a closed-loop position-loss failure. It does not establish that more
updates alone will solve the task, nor that support supervision helps. If
the fixed final trial still fails, analyze the contact/reference transition
near failure using saved native contacts/actions/geometry and complete pose
errors before changing the next experiment. Do not retrofit this trial's
gates, reset behavior, reference, physics, budget or checkpoint selection.

All original research-plan-v1.18 requirements remain active. World models
still lack Actor-gradient qualification. Long training continues;final control
success, real ego retargeting, multitrajectory/seed amortization and hardware
deployment remain unproven.
