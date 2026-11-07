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
