"""Strict three-field observation reader and source-bound catalogs.

No cached wrist, robot trajectory, contact/force/phase label, network clipID,
robot FK, optimization, simulation, or training is introduced here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np

FIELDS = frozenset(("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
CATALOG_KEYS = frozenset(("schema", "track", "partition", "allowed_uses", "lineage", "split_group",
                          "upstream_bundle", "upstream_inventory", "entries"))
ENTRY_KEYS = frozenset(("source_id", "source_file", "source_sha256", "export_file", "export_sha256",
                        "frame_range_half_open", "partition_role"))
CATALOG_SCHEMA = "shared_three_field_observation_catalog:1"
TRACK_USES = {
    ("inventory_audit", "all"): ("geometry_preparation", "inventory_audit"),
    ("controlled_within_lineage_holdout", "train"): ("geometry_preparation", "train"),
    ("controlled_within_lineage_holdout", "validation"): ("geometry_preparation", "validation"),
    ("controlled_within_lineage_holdout", "test"): ("geometry_preparation", "test"),
    ("seen_reference_cost_curve", "train_seen_eval"): ("geometry_preparation", "seen_eval", "train"),
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA256")
    return value


def _checked_file(path: str | Path, expected: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Catalog contract paths must be absolute")
    path = path.resolve(strict=True)
    if not path.is_file() or sha256_file(path) != _digest(expected):
        raise ValueError("Pinned file missing or SHA256 mismatch: " + str(path))
    return path


def _pin(pin: Any) -> Path:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("Expected strict path/SHA256 binding")
    return _checked_file(pin["path"], pin["sha256"])


def _json(path: Path) -> dict:
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=nonfinite)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping): return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple): return [_thaw(v) for v in value]
    return value


def _array(value, *, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    # A bytes backing buffer also rejects setting WRITEABLE=True later.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _range(value, total: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int for v in value):
        raise ValueError("Frame range must contain two integers, not phases/booleans")
    lo, hi = value
    if lo < 0 or hi <= lo or hi > total:
        raise ValueError("Frame range outside frozen source")
    return lo, hi


def validate_observation_arrays(arrays: Mapping[str, np.ndarray], *, frames: int, frame_dt_s: float) -> None:
    if set(arrays) != FIELDS:
        raise ValueError("Only time/hand landmarks/object pose fields are allowed; labels/wrist/phase forbidden")
    t, hand, pose = (np.asarray(arrays[k]) for k in ("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
    if type(frames) is not int or frames < 2 or not np.isfinite(frame_dt_s) or frame_dt_s <= 0:
        raise ValueError("Invalid frozen frame count/dt")
    if t.shape != (frames,) or hand.shape != (frames, 21, 3) or pose.shape != (frames, 7):
        raise ValueError("Observation/time shape differs from source inventory")
    if any(a.dtype.kind != "f" or not np.isfinite(a).all() for a in (t, hand, pose)):
        raise ValueError("Observations must be finite floating arrays")
    if not np.all(np.diff(t) > 0) or not np.allclose(t, np.arange(frames)*frame_dt_s, rtol=0, atol=1e-12):
        raise ValueError("Source time must preserve the pinned 50Hz frame clock; no phase/time warp")
    if np.max(np.abs(np.linalg.norm(pose[:, 3:], axis=-1)-1)) > 1e-6:
        raise ValueError("Object wxyz quaternion not unit within1e-6; no silent normalization")


def read_three_field_export(path: str | Path, *, expected_sha256: str, frames: int, frame_dt_s: float):
    """Low-level array contract, used after source-bound catalog validation."""
    path = _checked_file(path, expected_sha256)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS or len(archive.files) != len(FIELDS):
            raise ValueError("Export field allowlist differs or duplicate keys found")
        arrays = {key: archive[key].copy() for key in FIELDS}
    validate_observation_arrays(arrays, frames=frames, frame_dt_s=frame_dt_s)
    return MappingProxyType({key: _array(value) for key, value in arrays.items()})


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    source_id: str
    source_file: Path
    source_sha256: str
    export_file: Path
    export_sha256: str
    frame_range_half_open: tuple[int, int]
    partition_role: str


@dataclass(frozen=True, slots=True)
class VerifiedCatalog:
    path: Path
    sha256: str
    track: str
    partition: str
    allowed_uses: tuple[str, ...]
    lineage: str
    split_group: str
    entries: tuple[CatalogEntry, ...]
    inventory_path: Path
    inventory_sha256: str
    bundle_path: Path
    bundle_sha256: str
    _rows: Mapping[str, Any] = field(repr=False)

    @property
    def source_ids(self):
        return tuple(entry.source_id for entry in self.entries)


@dataclass(frozen=True, slots=True)
class ObservationTrajectory:
    time_s: np.ndarray
    hand_landmarks_source_w: np.ndarray
    object_pose_source_w: np.ndarray
    provenance: Mapping[str, Any]

    def provenance_dict(self) -> dict:
        return _thaw(self.provenance)


def open_catalog(path: str | Path, *, expected_sha256: str) -> VerifiedCatalog:
    path = _checked_file(path, expected_sha256)
    c = _json(path)
    if set(c) != CATALOG_KEYS or c["schema"] != CATALOG_SCHEMA:
        raise ValueError("Catalog schema/allowlist differs")
    key = (c["track"], c["partition"])
    if key not in TRACK_USES or tuple(c["allowed_uses"]) != TRACK_USES[key]:
        raise ValueError("Unrecognized track/split/use contract")
    bundle_path = _pin(c["upstream_bundle"])
    bundle = _json(bundle_path)
    if bundle["schema"] != "shared_multitrajectory_controlled_bundle:1" or bundle["GPU_authorized"]:
        raise ValueError("Expected frozen CPU draft source bundle, never execution authority")
    bound_files = {k: _pin(v) for k, v in bundle["artifacts"].items()}
    inventory_path = _pin(c["upstream_inventory"])
    if inventory_path != bound_files["inventory"] or c["upstream_inventory"] != bundle["artifacts"]["inventory"]:
        raise ValueError("Catalog/inventory binding differs from frozen bundle")
    inventory, protocol = _json(inventory_path), _json(bound_files["main_protocol"])
    if inventory["source_count"] != 10 or len(inventory["rows"]) != 10 or inventory["verified_independent_ego_demonstrations"] != 0:
        raise ValueError("Expected10 controlled clips with zero verified independent ego demos")
    rows = {row["source_id"]: row for row in inventory["rows"]}
    if len(rows) != 10 or len({r["lineage"] for r in rows.values()}) != 1 or len({r["split_group"] for r in rows.values()}) != 1:
        raise ValueError("Conservative single lineage/split_group required")
    if c["lineage"] != next(iter(rows.values()))["lineage"] or c["split_group"] != next(iter(rows.values()))["split_group"]:
        raise ValueError("Catalog attempts to relabel lineage/split_group")
    _pin(inventory["source_catalog"])
    if key[0] == "controlled_within_lineage_holdout":
        split = protocol["data_tracks"]["controlled_within_lineage_holdout"]
        eligible = set(split["training_pool" if key[1] == "train" else key[1]])
    elif key[0] == "seen_reference_cost_curve":
        eligible = set(protocol["data_tracks"]["seen_reference_cost_curve"]["prefix_order"])
    else:
        eligible = {s.rsplit("/", 1)[1] for s in rows}
    if not isinstance(c["entries"], list) or not c["entries"]:
        raise ValueError("Catalog must enroll at least one trajectory")
    entries = []; enrolled = set()
    for e in c["entries"]:
        if set(e) != ENTRY_KEYS or e["source_id"] not in rows or e["source_id"] in enrolled:
            raise ValueError("Unknown/duplicate source or forbidden catalog label field")
        row = rows[e["source_id"]]
        if e["source_id"].rsplit("/", 1)[1] not in eligible or e["partition_role"] != c["partition"]:
            raise ValueError("Source is forbidden in this catalog partition (external009/010 cannot train)")
        if e["source_sha256"] != row["source_sha256"] or Path(e["source_file"]).resolve() != Path(row["source_file"]).resolve():
            raise ValueError("Source path/SHA differs from pinned source inventory")
        export = row["observation_export"]
        if e["export_sha256"] != export["sha256"] or Path(e["export_file"]).resolve() != Path(export["path"]).resolve():
            raise ValueError("Export path/SHA differs from pinned observation inventory")
        source = _checked_file(e["source_file"], e["source_sha256"])
        observed = _checked_file(e["export_file"], e["export_sha256"])
        bounds = _range(e["frame_range_half_open"], row["frames"])
        entries.append(CatalogEntry(e["source_id"], source, e["source_sha256"], observed, e["export_sha256"], bounds, e["partition_role"]))
        enrolled.add(e["source_id"])
    return VerifiedCatalog(path, expected_sha256, *key, TRACK_USES[key], c["lineage"], c["split_group"], tuple(entries),
                           inventory_path, c["upstream_inventory"]["sha256"], bundle_path,
                           c["upstream_bundle"]["sha256"], _freeze(rows))


def load_observation(catalog: VerifiedCatalog, source_id: str, *, use: str,
                     frame_range: tuple[int, int] | None = None) -> ObservationTrajectory:
    if not isinstance(catalog, VerifiedCatalog) or use not in catalog.allowed_uses:
        raise ValueError("Use is not authorized by this data catalog; this is not GPU/training authority")
    _checked_file(catalog.path, catalog.sha256)
    _checked_file(catalog.inventory_path, catalog.inventory_sha256)
    _checked_file(catalog.bundle_path, catalog.bundle_sha256)
    selected = [entry for entry in catalog.entries if entry.source_id == source_id]
    if len(selected) != 1: raise ValueError("Source not enrolled in this catalog")
    e = selected[0]; row = catalog._rows[source_id]
    _checked_file(e.source_file, e.source_sha256)
    arrays = read_three_field_export(e.export_file, expected_sha256=e.export_sha256, frames=row["frames"], frame_dt_s=row["frame_dt_s"])
    lo, hi = e.frame_range_half_open if frame_range is None else _range(frame_range, row["frames"])
    if lo < e.frame_range_half_open[0] or hi > e.frame_range_half_open[1]:
        raise ValueError("Requested slice exceeds frozen catalog enrollment")
    provenance = {
        "schema": "three_field_observation_provenance:1", "source_kind": "upstream_kinematic_controlled_cached",
        "source_id": source_id, "source_file": str(e.source_file), "source_sha256": e.source_sha256,
        "export_file": str(e.export_file), "export_sha256": e.export_sha256,
        "catalog_file": str(catalog.path), "catalog_sha256": catalog.sha256,
        "inventory_sha256": catalog.inventory_sha256, "upstream_bundle_sha256": catalog.bundle_sha256,
        "source_catalog_sha256": row["source_catalog_sha256"], "source_frame_range_half_open": row["source_frame_range_half_open"],
        "catalog_frame_range_half_open": e.frame_range_half_open, "selected_source_frame_range_half_open": [lo, hi],
        "source_frame_dt_s": row["frame_dt_s"], "source_times_preserved_without_rebase": True,
        "lineage": catalog.lineage, "split_group": catalog.split_group, "track": catalog.track,
        "partition": catalog.partition, "use": use, "conservative_within_lineage_role": row["trajectory_role"],
        "confidence": _thaw(row.get("confidence", {"status": "unknown", "values": None, "reason": "No landmark/object confidence array in the pinned3field export"})),
        "qualifications": {"is_real_ego": False, "independent_ego_demonstrations_verified": 0,
                           "camera_frame": "unqualified", "metric_scale": "unqualified", "handedness": "unqualified",
                           "derived_robot_reference": False, "physical_reference": False, "teacher": False},
        "boundary": {"read_fields": sorted(FIELDS), "source_robot_q_link_wrist_used": False,
                     "cached_wrist_created": False, "contact_phase_force_labels_created": False,
                     "source_id_is_network_input": False, "training_or_GPU_authorized": False},
        "ABI_warning": "Observation21 are sourcehand landmarks. Palmroot macroFK21 differs from native21 BODYlink targets; downstream derivedreference exporter must recompute exactRIGHT_HAND_LINK_NAMES from derivedq, never rename a macroFK array.",
    }
    return ObservationTrajectory(*(_array(arrays[key][lo:hi]) for key in
                                   ("time_s", "hand_landmarks_source_w", "object_pose_source_w")), _freeze(provenance))


def geometry_arrays(observation: ObservationTrajectory):
    """Thin in-memory input to existing hand_frames/directions; no wrist filler.

    Existing seven-field CLI consumers stay unchanged. A new geometry caller
    can use these three arrays directly and instantiate the pinned RobotKin.
    No robot reference, link ABI, confidence or physical qualification follows.
    """
    if not isinstance(observation, ObservationTrajectory): raise TypeError("Expected verified ObservationTrajectory")
    return MappingProxyType({name: _array(getattr(observation, name), dtype=np.float64) for name in FIELDS})
