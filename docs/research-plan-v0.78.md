# Recheck visited-source prediction before extending native planning horizon

Previous turn is PROGRESS: eight actual velocity-objective cases completed;
the final metadata failure was recovered explicitly without new physics,
independent review/tick diagnosis retained two replay failures (132/134 pass) and lost
import-list qualification, and203 tests pass. Full Goal remains ACTIVE.
All tasks still fail; velocity costs do not improve survival. No weight sweep,
Actor update, qualified expert or true ego result follows.

The largest new mismatch is5ms velocity+support at source180ms/control9:
first three physics ticks pass, fourth has0.464814N force,0.038332rad rotation
and0.077469m/s velocity error despite exact source/control/clock checks.
The fine-dt state-only source20ms/control1 first fails force at15ms elapsed.
Earlier calibration at sources0–120ms does not cover the new180ms source.

1. Predeclare diagnostic anchors covering both failed sources plus matched
   passing source/control states from the SAME saved episodes (e.g adjacent
   controls and the common initial source). Use all anchors, not favorable
   selection. Preserve original fixed model, numerical policy, dt, actual
   selected absolute PD targets and20ms clock. No new gains/material changes,
   loss sweeps, expanded horizon or training while this fidelity is unresolved.
2. Reconstruct independent one-world model/Data from saved topology, all404
   model arrays/195 scalars and169 source arrays. Require exact model hashes,
   array keys/shapes/dtypes and full source byte restore (including unused
   scratch NaNs). Old native-controller packets omitted dynamic Data scalar
   JSON: any capacity/layout values derived from shapes/type declarations or
   fresh reconstruction must be labeled reconstructed metadata. They must
   never be called original observed scalars or silently inferred from a
   different experiment. Reject incompatible layouts rather than truncate.
3. Capture full reconstructed Data scalars/layout before diagnostics, and save
   original observed Data scalar/layout JSON in EVERY future fresh native
   source capture. Preserve immutable source receipts and the executed-import
   enumeration repair. Failed collection/recovery and failed initial historic
   review remain unchanged with explicit provenance; no clean rerun just to
   erase those records. Driver/CUDA/installed packages stay untouched.
4. Repeat the identical held-PD20ms execution from each saved source at least
   three times; independently compare every tick to original selected
   forecast and actual execution at original state/ctrl/time/force gates.
   Save pre/post solver counters, active contact/constraint labels and force
   decoding, warmstarts and capacity/no-aid checks for ALL repeats. Repeat
   spread is descriptive, not a statistical confidence interval. Record full
   native work and any reconstruction-only limitations. No live episode reset
   or assistance is introduced; these are DIAGNOSTIC_NEVER_FIT counterfactuals.
5. If divergence persists, distinguish source reconstruction, execution path,
   numerical repeatability and contact-mode sensitivity using explicit matched
   evidence; do not claim solver/contact cause from failure timing alone.
   Add an independent actual collection only if needed to disambiguate paths,
   with original state/command read-only around probes. Do not relabel failed
   prefixes as experts or qualify learned gradients from a handful of points.
6. Only after relevant short-horizon fidelity is understood consider a longer
   native prediction with correct20ms reference/EMA/PD advancement and fresh
   sequence/budget validation. A held-PD60ms fork is a different controller;
   current20ms results do not pre-qualify it. Full-orientation/axial-roll
   mismatch also remains a separate unmet unified task requirement.

This diagnostic is preparation for contact-correct learned control, not a
replacement for original SuperMimic/CHORD/save-resume, cube/pen evaluation,
fair shared multi-trajectory cost, world-model gradient qualification, true
ego assets/dynamics/input contracts or hardware observations. Full Goal stays
ACTIVE, no current training/checkpoint process waits, no new pause request.
