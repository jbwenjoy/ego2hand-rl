# Derived collision reproducer identifies overflow and silent missing contacts

No training or integration was added. Original long/recovery failures and
their physical/source/budget contracts remain unchanged. All new GPU/CPU
queries use explicitly isolated diagnostic fixtures, never NOT_RESUMABLE
payloads as a training restore. Goal remains ACTIVE/incomplete.

## Geometry evidence and isolated pair

Original failed252 world31 rows61/62/63 satisfy exact current-policy versus
previous-terminal bytes, time and phase alignment. Source_R/C recomposed
features differ by <=5.9604645e-8. The retained direct hand joint indices7..26
match zero encoder bias/default. Object quaternion is DERIVED by FP64 matrix
projection and FP32 cast; its unavailable original quaternion bytes are not
claimed recovered. COM is at free-body origin with fixed wrist/physical Model.

Fresh fixture404 Model arrays/195 scalars and169 Data arrays/8 scalars match
the valid parent completely; no scalar reconstruction override was needed.
Known-parent raw FK reproduces all geom poses/matrices byte-for-byte. Derived
parent error is max5.9604645e-8m position/2.3841858e-7 rotation entry. These
measure reconstruction precision, not original incident identity.

Unchanged installed full collision on5 registered scenes yields no overflow
for the two parent controls or original rows61/62; row63 DERIVED yields raw
world31 EPA_HORIZON256 and the horizon24 warning. A complete broadphase replay
then isolates ALL12 world31 candidates. Exactly one physical pair fails:
geom1 `object/cube` (CYLINDER5), geom55
`robot/right_finger4_link4_col` (MESH7), pair/sensor IDs[-1,-1]. Broadphase
atomic output indices differ311/322 across runs; stable identity is the ordered
geom IDs/world/pair IDs, never a chosen transient slot.

## Capacity alone is rejected

For all12 scenes (base+8 one-ULP quaternion neighbors+3 controls), source
copies change ONLY the unique EPA horizon allocation from24 to48/96. Source
headers are preserved; installed package bytes are unchanged. All raw results
and sorted contact bytes remain available, including failures.

|Group|Installed24|Isolated48/96|
|---|---|---|
|Base+8 neighbors|4/9 overflow256 and0 contacts;5/9 return1|0 overflow; SAME4/9 return0 contacts|
|Parent/row61/row62 controls|3/3 return1, no overflow|3/3 return1, output bytes match24|

48 and96 contact bytes agree for every scene, including EMPTY invalid cases.
Empty contacts being finite/equal does not establish correct collision.
Independent native FP64 `mj_geomDistance` on the SAME FP32 transforms and
physical sizes finds base penetration13.5600122 micrometers; all9 neighbors
remain penetrated. Iterations50/200 agree for all12 scenes. Native algorithm/
precision is a diagnostic reference, not ego force truth or an exact original
incident state. Capacity48/96 have NOT qualified for production/training.

## Translation is also insufficient

Using unchanged installed24, translating all world31 geom centers to geom1
origin or pair midpoint alters relative positions by <=9.31322575e-10m.
Native distance still indicates overlap for all36 cases. WORLD has4/12
overflow/missing contacts; each local placement removes overflow but has1/12
SILENT missing contact (different neighbor in each placement). Valid GPU
contact distances differ from native reference by at most0.541 micrometers.
This supports numerical sensitivity; it does not admit a pair-local-frame fix.

## Observed EPA branch

Four queries use isolated GJK/convex source copies with ONLY printf branch
observations, explicit24/48 scratch and import redirection. All4 match original
raw masks/contact counts. Actual logs show:

- Base24: horizon overflow and invalid EPA result, no contact.
- Base48: `_attach_face` returns0, degenerate face at nface256/horizon34,
  followed by idx=-1 invalid result. Raw overflow stays0 and no contact.
- Local24 missing neighbor: no logged degenerate-face/invalid-_epa branch;
  still raw0/contact0. Its earlier GJK/EPA-entry path remains untraced.
- Parent24 control:1 contact, no overflow/invalid branch.

This demonstrates why increasing a buffer can hide a failure without restoring
contact, for the DERIVED fixture. It does not prove original SUPPORT206 had
the same cause, nor that every missing-contact case shares this branch.

## Retained engineering failures and work

CapacityV1 exited1 during artifact serialization after1 full+12 isolated+
1 capacity query. Mixed scene responses caused a duplicate NPZ keyword. V2
uses a finite-FP32 geometry-only input reader and new registration/output.
All V1 frozen source, counters and partial evidence remain; its last query
output was not persisted and is not invented.

Native-distance reviewV1 failed BEFORE its query loop because shared leading1
geom_dataid was indexed as world31. V2 uses the engine's modulo batch indexing.
Original console/frozen source remain; pre-query status is reconstructed from
that source/trace, not a lost runtime counter. V1 wall is not known, not zero.

Actual added work across this milestone:7 full32-world collision/narrowphase
queries (224 world queries),101 isolated pair queries,60 native distance
queries,13 full32-world FK builds (416 world FK),0 integrations/Adam. Include
compilation, failed persistence/preflight and evidence overhead separately;
this is not a training speedup measurement or all historical research cost.
Full445 tests pass9.652s, including10 new tests for frame/zero-bias admission,
source-copy capacity isolation and mixed artifact handling.

## Next required action

Trace GJK/epa_phase entry/exit for the silent local case and identify its
termination reason. Keep false negatives visible; no finite/zero-overflow
shortcut may qualify a backend. Then compare explicitly pinned robust collision
variants against the failing scene, all ULP/translation/positive controls and
independent distance. An invalid-status guard is necessary evidence protection,
not itself a physical repair. Do not silently replace collider geometry or
force/contact data. Broader scene/contact/state/gradient qualification and a
fresh matched training/evaluation protocol are required before restarting.

All original M0–M5, controlled/real ego contracts, WM Actor gradient admission,
multi-reference/seed total-cost evaluations, source gravity/scale/inertia/
friction/RGB and real hardware subversion/zero/gains/latency/observation checks
remain. Current jobs are terminal; no new video or successful task is asserted.
