# Validate the local CCD-tolerance candidate on matched dynamic sources

Previous turn is PROGRESS: native broadphase contains pair1/29 in all20
recorded queries; separate GJK/EPA wrapper remains UNQUALIFIED (130/160 mixed
checks), cache-contamination hypothesis unsupported. Native-only2x2 CCD
factorial shows tolerance1e-7 yields contact for both recorded pose images,
while cap100 alone does not. Independent676/676 and206 regression tests pass.
ZERO new integrations/learning occurred. Full Goal remains ACTIVE/incomplete.

1. Candidate COLLISION numerical profile: CCD cap50 and device tolerance1e-7.
   Compare fresh baseline50/1e-6 on the SAME seven saved source anchors and
   complete57-action tables, three repeats per path. No further tolerance/
   iteration/gain sweep. Original geometry/material/mass/inertia/actuator and
   constraint-solver settings and all task/replay/EFC gates remain fixed.
   Explicit full model-field diffs must show only `opt.ccd_tolerance` changes.
2. Reuse one maintained saved-source/native-planning implementation. Each
   profile uses matched full public source arrays, original PD table/order,
   reference/EMA/score inputs and20ms horizon. Retain all57 predictions, all
   three repeats and instrumented held-PD counterparts. Equal anchors/actions/
   repeats gives equal integration budget; report compile/readback costs
   separately and do not call this production acceleration or learned A1/A2.
3. Compare within-profile repeated selected/all57 forecasts and native graph
   versus instrumented native execution using original state/ctrl/time/force
   gates. Independently decode every instrumented EFC wrench and momentum.
   New-profile versus OLD actual/forecast differences are numerical-model
   response differences, not same-model replay errors or repaired old records.
   Do not use the unqualified separate GJK/EPA wrapper for physical labels.
4. Original Data scalar JSON/import enumeration is unavailable; any rebuilt
   static capacities remain explicitly reconstructed, with exact array-layout/
   byte checks and fresh diagnostic metadata. Snapshots of all active and
   installed source bytes and numerical before/after receipts must stay exact.
   New fresh actual episodes record observed Data scalars/layout/imports;
   never backfill old runs or erase failed setup/stage reports.
5. Treat the candidate as fidelity-qualified for these saved sources only if
   ALL within-profile selected repeat/path gates, all57 repeat gates, source
   contracts and EFC/momentum checks pass. Otherwise retain an unqualified
   diagnostic profile; no global physics/gradient or expert promotion follows.
   A fresh full-clock no-aid controller probe may then test actual task behavior
   under the declared profile with all uncertainty explicit, using matched
   control objectives/candidate budgets and per-source forecast/actual audit.
   Retain all original COM/full-orientation/duration/geometry/force gates,
   first-terminal stopping and no object/reset assistance. Failed prefixes
   remain nonexpert. A single saved pose or contact-present flag cannot
   qualify broader states, world-model gradients or a stable learned policy.

This is a development numerical candidate chosen from two pose images, not
calibrated real dynamics or a held-out ego/hardware result. All original
SuperMimic/CHORD/save-resume, fair policy/multi-trajectory budget comparisons,
cube/pen evaluation, contact/gradient verification, ego asset/input/dynamics
and hardware contracts remain required. No active training/checkpoint wait,
no new pause request; Goal ACTIVE.
