# Research plan v1.24 — numerical false negatives, no capacity-only repair

Keep the entire original M0–M5 objective and v1.23 requirements active.

1. Derived geometry from original252 aligns across policy/time/phase. Parent
   raw FK is byte-exact; derived roundtrip position/matrix errors are measured.
   Original raw object quaternion is still unavailable; use DERIVED scope.
2. Unchanged full collision reproduces256 on row63. All12 candidate pairs
   isolated; sole failing pair world31/geoms[1,55] = cylinder object vs hand
   finger4_link4 mesh. Transient broadphase indices are not stable identity.
3. Capacity24/48/96 and9 ULP scenes plus3 controls completed. Larger buffers
   clear flags but retain4/9 zero-contact failures. Native FP64 distance
   indicates overlap (base13.56um). Translation removes flags but each local
   mode still loses one contact. NO capacity or translation variant admitted.
4. printf-only isolated trace4/4 flags/counts match. Base48 reaches degenerate
   `_attach_face` and invalid EPA return WITHOUT overflow flag. Silent local
   neighbor did not reach this logged branch; cause incomplete. NEXT trace
   GJK/epa_phase entry/termination for this case, then qualify explicit robust
   variants with invalid-result checks, contact/distance evidence and full
   positive/ULP/translation controls. No source/physics/state rebinding.
5. Preserve failed capacityV1 persistence (1full+13pair queries), zero-query
   native-distanceV1 preflight and all frozen source/counters. This milestone
   adds7 full collision batches,101 pair queries,60 distance queries,13 FK
   batches,0 integrations/Adam. Do not charge old work twice or call0native
   zero compute. Full445 tests pass; no training alive, no new policy video.
6. Only qualified collision behavior may enter a newly registered state/control
   backend, matched training and final nine-case LIVE W1 execution. Preserve
   full SO3/COM/geometry/no-aid/control/drive gates, native/causal/failed total
   cost, genuine A1/A2 and world-model gradient gates. Do not fit invalid data.
7. No WM Actor admission, task/ego/generalization/speedup claim. Continue all
   controlled cube/pen and real-ego mesh/frame/gravity/inertia/friction/RGB,
   multi-trajectory/seed and hardware contract work. Installed package/driver/
   CUDA bytes unchanged. Goal ACTIVE/incomplete, not blocked or completed.

Read pen-retained-collision-and-epa-results-2026-11-16.md for scopes/evidence.
