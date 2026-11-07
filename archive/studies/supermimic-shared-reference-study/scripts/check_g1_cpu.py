"""Real10export and negative contract checks for the G1 reader; CPU only."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_reference import open_catalog, load_observation, geometry_arrays, read_three_field_export

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ReaderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read(ROOT / "outputs/g1_reader_v1/manifest.json")
        cls.catalogs = cls.manifest["catalogs"]
        cls.audit = cls.open("audit_all10")
        cls.example = load_observation(cls.audit, "01_single_axis/clip_007", use="inventory_audit")

    @classmethod
    def open(cls, name):
        pin = cls.catalogs[name]
        return open_catalog(pin["path"], expected_sha256=pin["sha256"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)

    def altered_catalog(self, name, change):
        value = read(self.catalogs[name]["path"])
        change(value)
        path = self.tmp / "catalog.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, sha(path)

    def export(self, arrays):
        path = self.tmp / "observations.npz"
        np.savez_compressed(path, **arrays)
        return path

    def sample_arrays(self):
        return {name: np.array(value, copy=True) for name, value in geometry_arrays(self.example).items()}

    def low_read(self, path):
        return read_three_field_export(path, expected_sha256=sha(path), frames=len(self.example.time_s), frame_dt_s=.02)

    def test_actual10_full_reads_and_unknown_confidence(self):
        count = 0
        for sid in self.audit.source_ids:
            obs = load_observation(self.audit, sid, use="inventory_audit")
            count += len(obs.time_s)
            p = obs.provenance_dict()
            self.assertEqual(p["confidence"]["status"], "unknown")
            self.assertIsNone(p["confidence"]["values"])
            self.assertFalse(p["boundary"]["cached_wrist_created"])
            self.assertFalse(p["boundary"]["source_robot_q_link_wrist_used"])
            self.assertFalse(p["qualifications"]["physical_reference"])
            self.assertEqual(p["lineage"], self.audit.lineage)
        self.assertEqual((len(self.audit.source_ids), count), (10, 3987))

    def test_source_time_preserved_and_selected_frames_recorded(self):
        obs = load_observation(self.audit, "01_single_axis/clip_007", use="geometry_preparation", frame_range=(200, 351))
        np.testing.assert_array_equal(obs.time_s, self.example.time_s[200:351])
        self.assertEqual((len(obs.time_s), float(obs.time_s[0]), float(obs.time_s[-1])), (151, 4., 7.))
        self.assertEqual(obs.provenance_dict()["selected_source_frame_range_half_open"], [200, 351])
        self.assertTrue(obs.provenance["source_times_preserved_without_rebase"])

    def test_immutable_arrays_and_provenance(self):
        with self.assertRaises(ValueError): self.example.hand_landmarks_source_w[0, 0, 0] = 0
        with self.assertRaises(ValueError): self.example.hand_landmarks_source_w.setflags(write=True)
        with self.assertRaises(TypeError): self.example.provenance["lineage"] = "changed"

    def test_external_holdout_never_trains_in_holdout_track(self):
        train = self.open("within_lineage_train8")
        self.assertEqual(len(train.entries), 8)
        self.assertNotIn("01_single_axis/clip_009", train.source_ids)
        self.assertNotIn("01_single_axis/clip_010", train.source_ids)
        with self.assertRaises(ValueError): load_observation(train, "01_single_axis/clip_009", use="train")
        for name, sid in (("within_lineage_validation009", "01_single_axis/clip_009"), ("within_lineage_test010", "01_single_axis/clip_010")):
            with self.assertRaises(ValueError): load_observation(self.open(name), sid, use="train")

    def test_seen_track_explicitly_allows_seen_train(self):
        seen = self.open("seen_cost_all10")
        for sid in ("01_single_axis/clip_009", "01_single_axis/clip_010"):
            obs = load_observation(seen, sid, use="train", frame_range=(0, 2))
            self.assertEqual(obs.provenance["track"], "seen_reference_cost_curve")
            self.assertFalse(obs.provenance["boundary"]["training_or_GPU_authorized"])

    def test_catalog_injection_of009_into_train_rejected_even_with_new_catalog_hash(self):
        source = read(self.catalogs["within_lineage_validation009"]["path"])["entries"][0]
        source["partition_role"] = "train"
        path, digest = self.altered_catalog("within_lineage_train8", lambda c: c["entries"].append(source))
        with self.assertRaisesRegex(ValueError, "forbidden"): open_catalog(path, expected_sha256=digest)

    def test_catalog_hash_and_source_binding_mismatch_rejected(self):
        with self.assertRaises(ValueError): open_catalog(self.catalogs["audit_all10"]["path"], expected_sha256="0"*64)
        path, digest = self.altered_catalog("audit_all10", lambda c: c["entries"][0].update(source_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "Source path/SHA"): open_catalog(path, expected_sha256=digest)

    def test_export_binding_cannot_be_redirected(self):
        path, digest = self.altered_catalog("audit_all10", lambda c: c["entries"][0].update(export_file=c["entries"][1]["export_file"]))
        with self.assertRaisesRegex(ValueError, "Export path/SHA"): open_catalog(path, expected_sha256=digest)

    def test_lineage_cannot_be_relabeled(self):
        path, digest = self.altered_catalog("audit_all10", lambda c: c.update(lineage="independent_ego_demo"))
        with self.assertRaisesRegex(ValueError, "lineage"): open_catalog(path, expected_sha256=digest)

    def test_phase_field_cannot_enter_catalog(self):
        path, digest = self.altered_catalog("audit_all10", lambda c: c["entries"][0].update(phase=0))
        with self.assertRaisesRegex(ValueError, "forbidden"): open_catalog(path, expected_sha256=digest)

    def test_label_injection_rejected_with_correct_export_hash(self):
        for field in ("reference_qpos", "link_pos_w", "wrist_pose_source_w", "contact_force", "phase", "trajectory_id", "landmark_confidence_known"):
            with self.subTest(field=field):
                arrays = self.sample_arrays(); arrays[field] = np.zeros((len(self.example.time_s), 1))
                with self.assertRaisesRegex(ValueError, "allowlist"): self.low_read(self.export(arrays))

    def test_shape_finite_quaternion_dtype_contracts(self):
        for variant in ("shape", "nan", "quaternion", "integer"):
            with self.subTest(variant=variant):
                arrays = self.sample_arrays()
                if variant == "shape": arrays["hand_landmarks_source_w"] = arrays["hand_landmarks_source_w"][:, :20]
                elif variant == "nan": arrays["object_pose_source_w"][0, 0] = np.nan
                elif variant == "quaternion": arrays["object_pose_source_w"][0, 3:] *= 1.01
                else: arrays["hand_landmarks_source_w"] = arrays["hand_landmarks_source_w"].astype(np.int64)
                with self.assertRaises(ValueError): self.low_read(self.export(arrays))

    def test_duplicate_or_warped_source_time_rejected(self):
        for variant in ("duplicate", "warp", "origin"):
            with self.subTest(variant=variant):
                arrays = self.sample_arrays()
                if variant == "duplicate": arrays["time_s"][1] = arrays["time_s"][0]
                elif variant == "warp": arrays["time_s"] *= 2
                else: arrays["time_s"] += .02
                with self.assertRaisesRegex(ValueError, "time"): self.low_read(self.export(arrays))

    def test_export_byte_tamper_rejected_before_array_load(self):
        path = self.export(self.sample_arrays()); expected = sha(path)
        with path.open("ab") as stream: stream.write(b"hash-tamper")
        with self.assertRaisesRegex(ValueError, "SHA256"): read_three_field_export(path, expected_sha256=expected, frames=len(self.example.time_s), frame_dt_s=.02)

    def test_slice_range_restrictions(self):
        for bounds in ((-1, 2), (0, 99999), (3, 3), (True, 3), (0., 3.)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError): load_observation(self.audit, "01_single_axis/clip_007", use="inventory_audit", frame_range=bounds)

    def test_three_field_thin_geometry_features_without_legacy_run(self):
        pin = self.manifest["thin_geometry_interface"]["legacy_source"]
        self.assertEqual(sha(pin["path"]), pin["sha256"])
        source = ast.parse(Path(pin["path"]).read_text(encoding="utf-8"))
        selected = [n for n in source.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BONES" for t in n.targets))
                    or (isinstance(n, ast.FunctionDef) and n.name in ("hand_frames", "directions"))]
        ns = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), pin["path"], "exec"), ns)
        for sid in self.audit.source_ids:
            obs = load_observation(self.audit, sid, use="geometry_preparation", frame_range=(0, 2))
            arrays = geometry_arrays(obs)
            self.assertEqual(set(arrays), {"time_s", "hand_landmarks_source_w", "object_pose_source_w"})
            self.assertTrue(all(a.dtype == np.float64 and not a.flags.writeable for a in arrays.values()))
            points = arrays["hand_landmarks_source_w"]
            self.assertEqual(ns["hand_frames"](points).shape, (2, 3, 3))
            self.assertEqual(ns["directions"](points).shape, (2, 15, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/g1_reader_cpu_checks_v1")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReaderContract)
    with (args.output / "test_output.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"status": "PASS_G1_DATA_READER_ONLY" if result.wasSuccessful() else "FAILED_CPU_CONTRACTS",
              "tests_run": result.testsRun, "failures": [str(x) for x in result.failures], "errors": [str(x) for x in result.errors],
              "checker_sha256": sha(__file__), "reader_sha256": sha(ROOT/"src/shared_reference/reader.py"),
              "manifest_sha256": sha(ROOT/"outputs/g1_reader_v1/manifest.json"),
              "CPU_check_wall_seconds": time.perf_counter()-started, "clock_scope": "testbody, imports excluded",
              "modules_not_imported": {name: name not in sys.modules for name in ("torch", "warp", "mujoco_warp", "mujoco")},
              "new_collision_queries": 0, "new_physics_steps": 0, "training_transitions": 0,
              "reference_generation_or_physics_qualified": False}
    assert all(report["modules_not_imported"].values())
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("errors","failures")}, indent=2))
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__": main()
