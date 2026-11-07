# v11 source-normal-aware mapping review

The minimal sound derivative keeps the v10 model, full hardware q8 ranges, remaining12q, phone boxes, two starts, algorithms and budgets. It changes a pinned anchor bundle and a new cheap-FK constructor; the frozen v9/v10 files stay intact. No model/FK/query/LP was run for this review.

## What the saved comparison establishes

|Finger|Source phone triangle|Old CAD triangle|Material displacement|Normal difference|
|---|---:|---:|---:|---:|
|Thumb|367|24|4.812745 mm|61.033892 deg|
|Index|6|139|4.738211 mm|58.008468 deg|

Both planar and near-coplanar group identities differ. Source B is an ego near-distance candidate projected onto the convex phone surface, not a measured contact label. A proper source branch replaces B, its filled triangle, outward normal and tangent basis together. Preserve exact original B bytes, and store the sub-roundoff difference from the triangle-projection reconstruction as evidence.

## Required hooks

1. `AssemblyKinematics(..., anchors_path=...)` currently still rejects any hash except307 and any schema except7. Make a genuinely separate helper v11 accepting an explicit expected pin/schema, with exact pin, thumb/index geom IDs, source pose, finite/unit normals and barycentric/outward checks. Do not monkeypatch frozen module globals. Keep its query equations and ownData behavior unchanged.
2. Runner must explicitly pass the new protocol anchor path/pin. Before all full evaluations/caches, set `e.anchors=kin.hand_points.copy()`, `e.anchor_targets=kin.phone_points.copy()`, `e.material=kin.phone_points.copy()`. E already computes world A and its J on the corresponding hand body at the full evaluation x.
3. Retain same-x q/A/B checks and cache whole static J/bias/PD operands. Cheap K owns a different MjData; it does not update E's data. Do not read E's live data after a cheap-only move. Conditional LP uses the new A/current nH/same-x J; final actual native points are independently regenerated as before.
4. Retain old14/87/4 contracts only as parent provenance. Add a narrow custom-anchor contract, new schema/pins, and truthful scope text replacing stale frozen307/CAD assertions. No frozen helper/source change is needed.

## Branches and hand facets

First `exact_source_original_hand` should use phone367/6 with old hand2332/1876, isolating only phone mapping. A second normal-aware branch can select actual filled hull facets of the same geom15/28 in a declared local region. At one fixed source pose compare actual world hand normals with either the negative source phone normal or a separately named inward material-line direction; these objectives differ. Pick the projected point on that selected triangle, retain barycentric/triangle/locality evidence, freeze deterministic ranking/ties before search, and never rotate a normal without its real triangle. A normal-only whole-sensor selection can move to a different side and is not a tactile-pad identification. If a1mm source tolerance is introduced, label it algorithmic relocation, not empirical measurement uncertainty; keep exact B and displacement separate. No qualifying local facet must be reported unavailable.

The catalog represents convex hull triangles of the compiled FP32 meshes. It does not establish raw scan fidelity, soft-pad front semantics, a unique finite contact patch, force capacity or native holding. Keep all22/189, unknown rejection, pad window, actual native friction/J/PD/strict-repeat gates unchanged. A geometric/source mapping fix can be tested; it is not itself a successful ego retarget or holding result.
