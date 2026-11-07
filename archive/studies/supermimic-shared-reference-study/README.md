# Shared-reference G1 data boundary

This new workspace implements the CPU observation reader/catalog stage only. It consumes the ten pinned observation-only exports from the frozen shared protocol. It does not generate a robot reference, import a cached wrist, run physics, create force labels, or train a shared model. Original single-clip and seven-field fixture consumers stay unchanged.

```python
import sys
sys.path.insert(0, "D:/workspace/supermimic-shared-reference-study/src")
from shared_reference import open_catalog, load_observation, geometry_arrays

# Use the explicit catalog SHA in outputs/g1_reader_v1/manifest.json.
catalog = open_catalog(catalog_path, expected_sha256=catalog_sha256)
observation = load_observation(
    catalog, "01_single_axis/clip_007",
    use="geometry_preparation", frame_range=(0, 151),
)
arrays = geometry_arrays(observation)
# arrays has exactly time_s, hand_landmarks_source_w, object_pose_source_w.
provenance = observation.provenance_dict()  # JSON-compatible deep copy.
```

`ObservationTrajectory` has the three same-named immutable arrays and a provenance mapping. `VerifiedCatalog` exposes `source_ids`, `entries`, `track`, `partition`, and `allowed_uses`. Slices preserve source timestamps and record their half-open source frame range; they never silently rebase time. Confidence values remain `None/unknown`; no zero-confidence arrays or wrist placeholders are invented. Pose quaternions must already be unit wxyz within 1e-6 and are not silently normalized.

Catalogs contain only source/export path and SHA bindings, frame enrollment, split/use and provenance identities. The loader checks the caller-pinned catalog hash, the upstream bundle and inventory, original source SHA, and export SHA. Three-field NPZ allowlists also reject robot q/link/wrist, contact, phase and clip-ID injection even when a modified NPZ's own hash is known. Catalog source IDs are lookup/audit metadata; `geometry_arrays` never includes an ID as a network feature.

The controlled within-lineage train catalog excludes clips009/010. Separate validation009 and test010 catalogs reject `use="train"`. The seen-reference cost track explicitly permits all ten references for training and seen evaluation; it must never be reported as held-out generalization. Audit-all10 permits observation inspection/geometric preparation, not training. Any future calibration fitted across trajectories must use its declared training split; reading geometry is not authorization to fit on held-out data. All clips share one conservative lineage/split_group; verified independent ego-demo N remains zero.

For the thin geometry connection, a new caller may pass these arrays to the pinned `hand_frames` and `directions` functions in `D:/workspace/supermimic-observation-retarget-study/scripts/retarget_palmroot_cpu.py`, instantiate its `RobotKinematics`, and preserve all provenance/uncertainty. The old seven-field `run()` interface intentionally rejects these three-field exports and is not modified. Do not add four dummy fields to trick it.

**The 21-point ABIs differ.** Source landmarks are observational points. Palmroot `fk_hand_points_w` uses palm plus link1/link3/link4/tip. Native `RIGHT_HAND_LINK_NAMES` uses palm plus link1/link2/link3/link4 bodies. A later native reference exporter must recompute the exact native BODY positions from derived q. Renaming a macro-FK array to `link_pos_w` is invalid. No native ABI export is claimed here.

Use the existing isolated Dex Python environment to run `scripts/build_g1_catalogs_cpu.py`, then `scripts/check_g1_cpu.py`. Both are CPU-only and refuse artifact overwrites. The manifest and tests record concrete source hashes and cost scope; CPU reader cost is separate from retargeting, physical qualification, teacher, pretraining, adaptation and evaluation costs. No shared speedup is assumed.
