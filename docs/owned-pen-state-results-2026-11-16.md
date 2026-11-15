# Owned native4 cache/contact state — 2026-11-16

The explicit owned runtime now integrates cached reference features and every-
native-step contact windows. Its native4 state restores833 array fields and804
scalar fields byte-exactly in same/fresh processes at1 and32 worlds, including
mixed windows after actual partial env resets. Final worker checks100/100,
independent review67/67 and318 tests pass. This is NOT a trained-policy checkpoint
or a physical future-repeatability qualification.

## Maintained components and ownership

The new files owned_pen_runtime.py / owned_pen_state.py delegate actions,
observations, rewards, terminations, resets, reference semantics and physical
configuration to existing maintained components. Existing native3, PPO training1,
ResidentContactMetrics and FrozenPenReferenceCache files remain unchanged.
OwnedPenFeatures only selects the qualified reference cache. The runtime tags
its separate owned runtime/transition contracts and native state schema
`controlled_pen_owned_state:4`; it does not pretend native3 saves this state.

The instance-owned Simulation.step hook resets the window immediately before
the first real native substep, then samples after EVERY2.5ms native step. Original
pending/shape guards run before that reset, so a rejected action preserves the
previous contact window. A normal20ms control collects8 samples. The additional
post-control forward is not mislabeled as a ninth native sample. The returned
contact_window dictionary contains BORROWED GPU views: the learner must copy
them before another sample/reset. This round does not modify PPO memory or train
with those views. Reward and termination rules remain the previous native ones.

Torch caller writes, Warp physics, contact metrics and consumers execute through
one owned Warp/Torch stream boundary, with event ordering to/from a different
caller stream. Graph/stream resources, process-local pointer bindings and the
completed last reset-ID upload are reconstructed resources. Mask/error buffers,
instantaneous/window outputs, source R/COM/clocks, counts, sticky flags and cache
tables ARE saved. Cache reference/mount/origin/table fingerprints and static
metric labels/joint-index arrays are checked before copying. Partial metric
reset preserves unselected histories even when env.reset performs a new forward.

Capture requires the existing historyless controlled observation contract.
Every native model/Data array, Python runtime leaf and alias is inventoried;
unknown mutable/opaque additions are rejected. Preflight checks schema, source/
engine/config/reference identity, field keys/types/shapes, immutable values,
expanded/overlapping saved views, and RNG formats before writes. RNG formats
are checked with private Python/NumPy/CPU/CUDA generators, not by installing
them globally. Native graph pointers/resources must remain owned. No physics
step occurs during source restoration. CUDA graphs are never pickled.

## Evidence and tests

| Final v6 process | Worker checks | Physical integrations |
|---|---|---|
|W1 producer|40/40|38|
|W1 fresh restorer|10/10|1|
|W32 producer|40/40|1216|
|W32 fresh restorer|10/10|32|
|Independent v3 review|67/67|0|

Producer sources include4 controls, a3-native-sample checkpoint, partial metric
reset plus2 diagnostic native samples, restore, and one future native sample.
The direct native calls are explicitly diagnostic fixed-PD sources; they are
not extra policy/control transitions. The4-control source and mid-window source
are both saved. Invalid schema, shapes, aliased bytes, cached/static values,
missing scalars, scalar types and RNG formats are rejected without changing
native/Python/RNG fingerprints. Replaced pointers, unknown runtime fields and
legacy capture are rejected; invalid pending actions preserve metric windows.

An actual partial env.reset is driven by CUDA IDs from a non-default caller
stream. Selected histories clear; unselected qpos/qvel/ctrl/time, reference
phase, EMA and all metric-window histories remain unchanged. Independently,
a fresh runtime restores this mixed-window partial-reset source with no
integration, and all saved native/Python/RNG bytes match the source. The review
also compares producer/fresh-restorer mid-window files field by field.

The13 new unit cases cover alias conflicts, exact NaN payloads, transposes,
Torch/Warp vector views, zero-stride world/scalar views, readonly arrays,
private RNG validation and failure without global RNG changes. Full suite318
passes. Source receipts freeze the exact executed code before each GPU run;
byte restoration does not substitute for physical rollout validation.

## Retained failures, iterations and cost

The initial W32 producer correctly exposed an implementation gap: expanded
Warp arrays had been validated but copying them was unsupported and failed
after earlier copies. That failed process was discarded;1184 integrations,
its source checkpoint, failure JSON and console remain. The final implementation
collapses shared zero-stride axes into an explicit destination view only after
checking identical saved values. Structural rejections are now checked before
copying and verified by before/after fingerprints. This is not a claim that
arbitrary external hardware failures are transactional.

Earlier positive W1/32 runs and all intermediate snapshots also remain, rather
than being overwritten after stream ordering, static-state or RNG validation
was strengthened. Across ALL worker versions the cost is6410 physical world-
integrations and0 optimizer updates. Final v6 processes account for1287 of
those; reviews add0. Earlier native3/PPO evidence and the previous cache profile
are retained under their own source versions and budgets.

## Future physics remains distinct

In final v6 the one-native-step fresh W1 future matches qpos/qvel/ctrl/time and
WORLD-COM wrench exactly. W32 qpos/ctrl/time match, but qvel differs by up to
3.8147e-6 and WORLD-COM wrench by2.81183e-8, with different contact ordering.
These are reported observations, not pass/fail substitutes for the earlier
longer W32 divergence. Source restoration is exact; deterministic physical
future replay remains unqualified and its root cause is not identified here.

No new policy weights, trained-state payload, strict full task success, WM
gradient, A1/A2 support advantage, ego generalization or training speedup is
claimed. The earlier~12.26% cache collection reduction excluded cold guards;
the integrated ownership/stream/guard path has not yet been benchmarked.

Compact evidence: provenance/owned_pen_state_results/ledger.json. Final runs:
owned_pen_W1/W32_producer/restorer_v6 and owned_pen_review_v3. All v1..v5 evidence
and failed W32 source remain under runs. Next step is an explicit training2
state/collector that copies borrowed contact windows before resets, followed
by declared geometry/support experiments under original strict gates.
