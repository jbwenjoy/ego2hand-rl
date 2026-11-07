# Research plan v0.40: leave the local contact pocket

2026-11-08. The preceding Goal turn was progress: native action authority and
checked static wrench bounds changed the next action. This turn implemented
and ran the new mechanically scored initialization experiment. Two seeds
passed initial full geometry but all four no-aid native cases failed. The
full Goal remains active, with no new learning/checkpoint/pause boundary.

## Established result, not completion

The native-force local optimizer stalled after tiny pose changes, despite a
larger declared trust box. Reduced score did not balance the phone. At the
original frozen contact pocket, even a relaxed35mm COM relocation cannot
balance gravity and moment inside the actual hand torque boxes, whereas
dropping object moment balance allows vertical forces above the phone's
weight. The difficulty is not captured by a scalar upward-force criterion.
These are local, numerical results; they prove neither global impossibility
nor that a new pose will work. Do not repeat the same optimizer with more
iterations or call its xtol termination successful grasp initialization.

## Next executable stage

1. Form explicit alternatives that move the contact patches/lever arms,
   rather than perturbing only micrometers inside the same native contact
   pocket. Derive the phone's long/tangent axes from the actual compiled
   collider, and predeclare a bounded sweep of phone sliding/pose seeds.
   Include q changes/contact-manifold restoration and force/moment/actuator
   checks. All seed attempts, absent contacts and geometry failures remain
   in the report. Use the shared mechanics implementation.
2. If a broader initialization trust domain is needed, label it a new
   development experiment and record full deviations from the original
   ego retargeted pose. The prior35mm/30degree bounds were algorithmic,
   not measured uncertainty. Do not change or reinterpret past outcomes,
   physics, fixed wrist, designated contacts or native holding gates.
3. A new candidate must receive fresh full22/189/table geometry checking
   and actual no-aid two-second evaluation at both dt. Static force-only
   witnesses cannot qualify it. Record contact migration, force margins,
   nonpad contact, timestep sensitivity and every missing tail.
4. If the bounded patch-moving search finds no useful candidate, use the
   existing ego intake/known-mesh inventory to choose a distinct object or
   clip with a more plausible reachable grasp. Preserve this phone failure.
   Do not redefine all of SuperMimic+CHORD around this contrived tripod task
   or claim that an altered contact-intent gate passes the original task.
5. On a mechanically useful route, rebuild the task/reference/observation/
   memory contract explicitly before Actor collection/training. Any rigid
   reference offset is reported, not inferred as true camera gravity.
   Then collect a fresh finite-action TRAIN intervention set; permanent
   heldout action responses and initialization development data stay out
   of fitting unless a separately declared new data collection is made.
6. Continue response-supervised WM/gradient validation and matched A1/A2
   policy comparison under the shared core. A changed task is a warmstart
   stage, never a resume of online300. Any new learning stage again stops
   after its first healthy checkpoint is verified and its ETA estimated,
   honoring the standing user-requested pause.

## Remaining full scope

Original controlled cube/pen reproduction and meaningful no-aid evaluation;
calibrated collision/inertia/friction robustness; independent ego hand/object
reconstruction, unknown gravity and contact-intent contracts; full61-frame
tracking; multi-trajectory sharing versus matched per-trajectory costs; and
hardware joint order/zero/gain/latency/observations all remain required.
A newly held nominal object alone cannot prove these, policy benefit or ego
generalization. Keep all historical files frozen and do not modify drivers
or CUDA. Complete the Goal only after its original requirements are proved.

Evidence: [initialization results](initialization-search-2026-11-08.md),
[prior authority audit](control-authority-2026-11-08.md), and
`provenance/initialization/summary.json`.
