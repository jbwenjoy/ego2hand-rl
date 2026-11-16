# CCD recovery candidate passes fixed static contact checks; dynamics not admitted

Goal ACTIVE/incomplete. No training/integration/Adam was added here. All old
failures, source/physics contracts and incurred costs remain. Installed package,
driver and CUDA sources were not edited. Candidate source is an explicit
isolated artifact, not an ordinary restore of an old native checkpoint.

## Corrections and distinct failure paths

The v1.24 phrase "degenerate face" was too specific: `_attach_face` returns0
both for projection failure and exhausted face storage. New printf-only trace
separates them and matches all4 reference masks/counts:

- Base24: EPA horizon overflow.
- Base48: INSIDE `_attach_face`, nface=capacity=256; no projection-failure
  print. This is unflagged face-buffer exhaustion, not proven geometric face
  degeneracy. Caller/inside-function counters differ, so no iteration count
  is inferred from the caller's printed counter.
- Silent local case: GJK dim4/dist0, EPA `_polytope4` status12 (the separate
  tetrahedron containment test disagrees with the four-point GJK simplex).
- Parent control: one contact, no error.

Upstream cache_kernel uses argument hashes and func.__name__, not module or
source. First same-process recovery/predicate studies retained builder names
after installed control populated cache. Their440 GPU query costs are real,
but ALGORITHMIC comparisons are cache-confounded and not admitted. Scratch
array capacities still changed at runtime and that evidence remains scoped.
Rerun uses8 namespaced factories and8 distinct kernel objects, recording module/
key/function bindings before queries and checking identity after every query.

That352-query rerun confirms:96/factor20 or40 scratch restores4 overflow cases;
closest-face retry and selected FP64 predicates still leave2 silent misses.
No variant from those runs qualifies as a complete repair.

## Bounded CCD direction retry

For cylinder/mesh ONLY, after an ambiguous EPA result, try at most3 fixed
orthogonal GJK initial directions with original geometry/support/cutoff/EPA.
Take the first finite negative-depth contact; preserve ALL raw flags. Original
GJK-separated path stays, no CPU fallback/force/contact injection is used.
Both factories are namespaced and their kernel identities checked.

44 scenes contain40 clear penetrations,3 clear separations,1 within1um boundary.
Retry restores all40 penetrations with no raw overflow/clear-separation contact;
19 EXTRA CCD helper calls are observed across9 queries. Compiler structure can
change which cases require retry; source-level branches alone do not establish
byte-equivalent execution. Every actual output and baseline difference is kept.

All11 filtered cylinder/mesh pairs x44 scenes then produce484 oracle cases:
175 OVERLAP,308 SEPARATED,1 BOUNDARY. Same saved FP32 shape sizes/transforms and
mesh vertices are independently queried by native FP64 mj_geomDistance.
Predeclared5um distance/5deg SIGNED normal/0.1mm midpoint criteria stay fixed.
Retry removes misses/false contacts but3 cases exceed5deg (max5.00444deg), so it
does NOT pass. The tolerance sweep1e-6/1e-7/1e-8 passes481/482/474 of484; maxima
5.004/6.648/32.749deg. Tighter precision alone is not accepted. Baseline1e-6
reproduces previous contact bytes for484/484 cases; oracle remains fixed.

## Same-face contact normal candidate

Original depth uses the chosen EPA face's projected-origin norm; normal uses
the difference of two separately weighted WORLD witnesses. The candidate
retains that chosen face index for cylinder/mesh and uses the SAME EPA face
projection for the contact frame. Static multicontact dispatch still excludes
cylinders. Witness midpoint, physical shape/materials,1e-6 tolerance,50
iterations,96/factor20 scratch, bounded retry and all raw flags are preserved.
No old state/source metadata is rewritten.

Two distinct kernels query all484 fixed cases each. Control matches previous
bytes484/484 and still fails3 normal checks. FACE_NORMAL passes484/484 under
the ORIGINAL criteria, with no overflow, nonfinite contact, clear-overlap miss,
or clear-separation false contact. Boundary remains explicitly separate.

|Candidate maximum|Observed|Predeclared limit|
|---|---:|---:|
|Signed depth error|0.541um|5um|
|Signed normal angle|0.944deg|5deg|
|Contact midpoint error|10.291um|100um|
|Frame orthogonality entry error|2.40e-7|1e-4|

Only the frame field changes bytes in175 nonempty contact responses; distance,
midpoint, identity, dimensions and raw flags match for all484 cases.309 empty
responses match completely.
This is an intentional collision-normal semantics change, not a proven
bit-preserving performance optimization. Native FP64 is a diagnostic reference,
not force ground truth. The scenes derive from the viewed controlled data;
they are not held-out real ego/generalization evidence.

## Work and validation

This milestone adds4272 GPU pair queries and528 native distance queries,
plus115 EXTRA CCD helper invocations (axis19,multifinger19,tolerance39,
face-normal comparison38). It includes cache-confounded440 queries.0 new FK,
native integration or Adam. Compilation, fresh fixture setup and evidence wall
are recorded per run;0 integrations does not mean0 compute or training speedup.
453 tests passed before contact-metric addition; full457 tests pass afterwards.
Extra retry helpers must not be silently folded out of later runtime costs.

The validated diagnostic source still prints retries. A quiet derivative is
different compiled source and needs requalification; never silently adopt it.

## Required next admission

1. Freeze this candidate and its exact cache namespace/source/normal/retry/
   numerical settings as an explicit backend identity. It is NOT yet eligible
   for original parent200/201 native resume. Old backend remains loadable.
2. Add new full-scene fresh-state step/forward/partial-reset tests with all
   primitive/convex paths, raw overflow/geometry/force/normal/impulse/no-aid
   checks and correct control timing/streams. Track every actual integration.
3. Qualify saved/resumed state and actor/norm/Adam/counters for the NEW backend.
   Runtime metadata must attest exact dynamic function bindings/source artifacts,
   so file pins alone cannot hide a Python hook or reused kernel. Rebinding
   old native metadata is forbidden; use fresh states/explicit admitted migration.
4. Re-evaluate dynamics/contact labels and gradient qualification under this
   backend. Old WM response data and gates belong to the old solver; do not
   silently mix them or promote an Actor. No model currently has admission.
5. Only then register fresh matched residual/support control training and the
   full strict LIVE W1 final evaluation/video. Keep all failed/prior costs and
   full original M0–M5, controlled/real ego/hardware/multitrajectory requirements.

No live training or new policy video. Full objective remains incomplete.
