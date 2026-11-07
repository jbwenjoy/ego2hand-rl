# One maintenance home

`src/supermimic_research/runtime` contains the current phone state, SO(3),
momentum, structured decoder, compiled geometry, directional descriptor, task,
reference and native-reader implementations. These modules use ordinary
package imports and explicit data paths. Historical JSON `sources` fields are
provenance, not selectors that import Python from another study directory.

`src/supermimic_research/learning` contains the world model, imagination and
shared WM/Actor/Critic update primitives. Offline and online experiments should
call these functions with explicit batches, models, normalizers and recipes.
Do not copy a trainer to change its reward mode or experiment budget.

`learning/online_runner.py` now orchestrates the same update primitives. Its
CPU `collection_worker` injects explicitly validated Actors into
`runtime/native_collection`; native labels and raw persistence use ordinary
package imports. `learning/checkpoint.py` supplies one shared raw readback
verifier, with stage-specific cursor/lineage checks in thin callers. The
offline command rejects online recipes. No old study source is imported by
the new training or collection path.

`src/chord_study` is the one maintained CHORD study package. Exact duplicate
packages from the C2 and penalty studies are retained only as historical
snapshots. `src/dex_supermimic` preserves the upstream namespace and notices;
the current phone task uses its Actor/Critic/normalizers, not the old PenMDP
state chart.

`src/supermimic_research/mechanics` is the maintained CPU initialization and
collision qualification implementation. It owns explicit model/data inputs,
uses ordinary imports, and has no dependency on old study Python paths. Its
new proposal-source closure is recorded separately from the frozen online300
runtime closure. Native reset generalized rotational force is about the free
joint origin in BODY axes; it must not be confused with the WM's moving-COM
wrench labels. A mechanically proposed pose is not an Actor-compatible task
until its reference/observation/memory bundle has explicitly been rebuilt.

`src/supermimic_research/assets` owns explicit source-frame baking, surface
inertial priors, compound XML and geometry validation primitives. Asset CPU
dependencies live in `.venv-assets`; the frozen research environment is not
reinstalled. A checked object-only asset has its own artifact manifest and
layout contract, and cannot stand in for a trained Hand2 task. Multiple convex
pieces must retain component identity and aggregate contacts consistently;
the visual mesh is never enabled for collision.

`ego/hand_frames.py` reads declared annotation windows and defines translated
rigid rest-palm frames. Its pinned Apache-2.0 UmeTrack vendor only reconstructs
human geometry; it does not provide robot joint targets. `assets/hand2_scene.py`
composes the maintained Hand2 factory and checked compound mug through MjSpec,
comparing actual named/body-local geometry identities instead of array prefixes.
Its output is a new task layout, not a compatible phone checkpoint resume.
`assets/proximity.py` caches an unsigned triangle query acceleration structure;
source-surface proximity remains distinct from contact truth or solid clearance.

`mechanics/compound_collision.py` qualifies declared object components and hand
sets using the shared convex-query primitives. `retargeting.py` and
`contact_ik.py` generate bounded kinematic proposals, with native witness
penalties distinguished from independent clearance. `compound_contacts.py`
sums each native contact once about the object's moving COM, including force
signs, local couples and canonical groups. `compound_holding.py` supplies
no-aid static-hold diagnostics with the frozen numerical gate values and an
explicit new mug contact-region mapping. It does not turn the mug layout into
a phone checkpoint resume or imply source-gravity/trajectory preservation.

`mechanics/support.py` builds native condim3 pyramidal force/COM-wrench and
hand-load columns, explicit affine-PD torque boxes and frozen static support
LPs. The saved reviewer independently compares them with native EFC unit
columns, accounting for BODY-root versus moving-COM moments. Its hypothetical
coefficients are never applied as simulator forces. `force_initialization.py`
searches bounded native initial pose/command proposals; optimizer termination
alone is not equilibrium or physical task qualification.

`mechanics/contact_modes.py` creates fresh, separately pinned pad/object spin
variants and checks full force/couple/COM/hand-load columns against native EFC
unit queries. It does not modify frozen models or calibrate material parameters.
`patch_intent.py` keeps finger-indexed distance/normal correspondence and unknown
source geometry explicit. Source skin/scan candidate envelopes are not observed
contact labels; whole-hand or per-finger unit-force similarity is diagnostic,
not dynamic task success or a validated reward.

`surface_patch_ik.py` tracks fixed collision-surface material points/normals with
native point/angular Jacobians; `collision_patch_ik.py` composes these terms with
the existing collision penalties rather than another copied solver. Candidate
source masks are hypotheses. `component_preload.py` makes small selected-pair
engagement proposals inside unchanged hard limits, with full audit afterward.
`wrench_separation.py` checks numerical separating inequalities of a declared
frozen6D ray cone with explicit moment scale. No source-contact truth, global
infeasibility, native dynamic success or physics gradient follows from them.

`contact_envelopes.py` builds unit-normal BODY candidate rays and rotates their
COM wrenches without changing tangent phase. These are not the CHORD unit-force
fingerprints or observed native contact manifolds. `contact_selection.py` adds
explicit three-group minimum coefficients to hypothetical triplet balance.
Selection, robot fit, actual native support and dynamic hold remain separate
gates; an actuator-box witness is never applied as a simulator contact force.

`configs` selects recipes and external artifacts. `ArtifactRegistry` resolves
named data roots, declared roles and content hashes. TRAIN, MONITOR, CHECK and
`ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE` remain different roles; a loader must not
glob all available NPZ files into fitting data.

`archive/studies` preserves original source/configuration/document bytes and
directory structure, including historically distinct experiment wrappers.
`provenance/workspace_inventory.json` maps imports to original paths and
hashes. Original Git HEAD/status/patches are recorded separately. Historical
code is not a second live implementation to update.

Large datasets, model assets, results, checkpoints, caches and environments are
external. Existing legacy processes and frozen absolute-path contracts remain
at their original locations. Archive import is a copy operation; it does not
claim those old directories can already be removed. In particular, directory
junctions do not preserve a contract that compares resolved absolute paths.

The archived experiments can be inspected for reproducibility. New development
must target the shared package and create configurations/run records in this
repository, rather than a new top-level `supermimic-*-study` folder.
