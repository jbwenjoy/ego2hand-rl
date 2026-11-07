"""Materialize source-bound data catalogs and read all10 existing exports, CPU only."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_reference import open_catalog, load_observation
from shared_reference.reader import _checked_file, _json, _pin, sha256_file

BUNDLE = Path("D:/workspace/supermimic-chord-study/protocols/shared-multitrajectory-controlled-v1-bundle.json")
BUNDLE_SHA = "f9cc02cef075ad22cf9a4ad68ed6dcc40f80056cc1efd5fb95691a2dba81a2e0"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/g1_reader_v1")
    args = ap.parse_args(); started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    bundle_path = _checked_file(BUNDLE, BUNDLE_SHA); bundle = _json(bundle_path)
    files = {name: _pin(pin) for name, pin in bundle["artifacts"].items()}
    inventory, protocol = _json(files["inventory"]), _json(files["main_protocol"])
    rows = {row["source_id"].rsplit("/", 1)[1]: row for row in inventory["rows"]}
    within = protocol["data_tracks"]["controlled_within_lineage_holdout"]
    definitions = [
        ("audit_all10", "inventory_audit", "all", list(rows), ["geometry_preparation", "inventory_audit"]),
        ("within_lineage_train8", "controlled_within_lineage_holdout", "train", within["training_pool"], ["geometry_preparation", "train"]),
        ("within_lineage_validation009", "controlled_within_lineage_holdout", "validation", within["validation"], ["geometry_preparation", "validation"]),
        ("within_lineage_test010", "controlled_within_lineage_holdout", "test", within["test"], ["geometry_preparation", "test"]),
        ("seen_cost_all10", "seen_reference_cost_curve", "train_seen_eval", protocol["data_tracks"]["seen_reference_cost_curve"]["prefix_order"], ["geometry_preparation", "seen_eval", "train"]),
    ]
    catalogs = {}
    for name, track, partition, ids, uses in definitions:
        entries = []
        for local in ids:
            row = rows[local]; exp = row["observation_export"]
            entries.append({"source_id": row["source_id"], "source_file": row["source_file"], "source_sha256": row["source_sha256"],
                            "export_file": exp["path"], "export_sha256": exp["sha256"],
                            "frame_range_half_open": row["source_frame_range_half_open"], "partition_role": partition})
        document = {"schema": "shared_three_field_observation_catalog:1", "track": track, "partition": partition,
                    "allowed_uses": uses, "lineage": entries and rows[ids[0]]["lineage"], "split_group": rows[ids[0]]["split_group"],
                    "upstream_bundle": {"path": str(bundle_path), "sha256": BUNDLE_SHA},
                    "upstream_inventory": bundle["artifacts"]["inventory"], "entries": entries}
        path = ROOT / "catalogs" / (name + ".json"); write(path, document)
        verified = open_catalog(path, expected_sha256=sha256_file(path))
        catalogs[name] = {"path": str(path), "sha256": sha256_file(path), "track": verified.track,
                          "partition": verified.partition, "count": len(verified.entries), "source_ids": list(verified.source_ids)}
    audit = open_catalog(catalogs["audit_all10"]["path"], expected_sha256=catalogs["audit_all10"]["sha256"])
    observations = []
    for sid in audit.source_ids:
        before = time.perf_counter(); obs = load_observation(audit, sid, use="inventory_audit")
        observations.append({"source_id": sid, "frames_read": len(obs.time_s), "source_first_last_time_s": [float(obs.time_s[0]), float(obs.time_s[-1])],
                             "read_wall_seconds_including_source_hash": time.perf_counter()-before,
                             "provenance": obs.provenance_dict(), "array_fields": ["time_s", "hand_landmarks_source_w", "object_pose_source_w"]})
    assert sum(x["frames_read"] for x in observations) == 3987
    read_manifest = output / "read_records.json"; write(read_manifest, {"schema": "G1_actual_read_records:1", "rows": observations})
    geometry_source = Path("D:/workspace/supermimic-observation-retarget-study/scripts/retarget_palmroot_cpu.py")
    native_constants = Path("D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py")
    manifest = {
        "schema": "shared_reference_G1_manifest:1", "status": "G1_READER_CPU_EXECUTED_NOT_REFERENCE_OR_PHYSICS_QUALIFICATION",
        "upstream_bundle": {"path": str(bundle_path), "sha256": BUNDLE_SHA}, "catalogs": catalogs,
        "source_code": {str(path.relative_to(ROOT)): {"path": str(path), "sha256": sha256_file(path)} for path in
                        [ROOT/"src/shared_reference/__init__.py", ROOT/"src/shared_reference/reader.py", Path(__file__)]},
        "actual_read_records": {"path": str(read_manifest), "sha256": sha256_file(read_manifest)},
        "implementation": {"strict_export_field_count": 3, "real_source_clips_read": 10, "frames_read": 3987,
                           "single_conservative_lineage": audit.lineage, "split_group": audit.split_group,
                           "independent_original_ego_demonstrations": 0, "cached_wrist_created": False,
                           "source_robot_q_link_wrist_used": False, "network_clipID_feature_created": False,
                           "confidence": {"status": "unknown", "values": None}, "old7field_fixture_consumer_modified": False},
        "thin_geometry_interface": {"API": "geometry_arrays(ObservationTrajectory) returns only3readonlyfloat64arrays; pass hand array to pinned hand_frames/directions, and instantiate RobotKinematics separately in a new geometry caller",
                                    "legacy_source": {"path": str(geometry_source), "sha256": sha256_file(geometry_source)},
                                    "native_link_constants": {"path": str(native_constants), "sha256": sha256_file(native_constants)},
                                    "native_ABI_warning": "Source21landmarks, palmrootmacroFK21(palm/link1/link3/link4/tip), and nativeBODY21(palm/link1/link2/link3/link4) are three different semantics. Derivedq must be re-FKed with exactnative names; no renaming."},
        "qualifications": {"G1_reader_stage": "implemented_CPU", "geometric_reference_batch": "not_implemented_here", "native_reference_ABI_export": "not_implemented_here", "physical_initialization": "not_qualified", "teacher_dataset": "not_created", "shared_training": "not_implemented_or_run", "real_ego": False},
        "cost_scope": {"stage": "CPU source/catalog validation +10export reads", "clock_start": "builderbody afterPythonimports", "started_utc": started_utc,
                       "builder_wall_seconds": time.perf_counter()-started, "per_read_wall_nested_in_builder": True,
                       "existing_exports_reused_not_regenerated": 10, "original_source_hashing_included": True,
                       "GPU_seconds": 0, "new_physics_steps": 0, "teacher_or_training_transitions": 0,
                       "scope_limits": "Not total projectwall, not amortizedsharedtrainingcost, no speedup claim"},
        "GPU_authorized": False,
    }
    write(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "catalogs": len(catalogs), "clips": 10, "frames": 3987, "manifest": str(output/"manifest.json")}, indent=2))


if __name__ == "__main__": main()
