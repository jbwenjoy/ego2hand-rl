import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from chord_study.acceptance import protocol_digest
from chord_study.reference import Reference

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("q_variant_generator", ROOT / "scripts" / "make_controlled_variants.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
BASE = ROOT / "outputs" / "pilot_rotating_reference_v2" / "reference.npz"
BUNDLE = ROOT / "outputs" / "reference_contact_targets_v1"


class QOnlyPerturbationAlgebra(unittest.TestCase):
    def test_first_two_samples_zero_then_smooth_ramp_and_constant(self):
        times = np.arange(151)*.02
        ramp = GENERATOR.quintic_bias_ramp(times)
        np.testing.assert_array_equal(ramp[:2], [0., 0.])
        self.assertGreater(ramp[2], 0)
        self.assertLess(ramp[10], 1)
        np.testing.assert_array_equal(ramp[11:], np.ones(140))
        self.assertTrue(np.all(np.diff(ramp) >= 0))

    def test_initial_q_qdot_and_actual_dtype_delta_are_exact(self):
        q = np.random.default_rng(8).normal(size=(151, 20)).astype(np.float32)
        directions = np.random.default_rng(20261102).choice([-1, 1], 20)
        changed, actual, requested, _ = GENERATOR.corrupt_q_target(q, np.arange(151)*.02, .03, directions)
        self.assertEqual(changed.dtype, q.dtype)
        np.testing.assert_array_equal(changed[:2], q[:2])
        np.testing.assert_array_equal(np.diff(changed[:2], axis=0)/.02, np.diff(q[:2], axis=0)/.02)
        np.testing.assert_array_equal(actual, changed.astype(float)-q.astype(float))
        np.testing.assert_array_equal(requested[:2], np.zeros((2, 20)))
        np.testing.assert_allclose(actual[-1], directions*.03, atol=2e-7)

    def test_out_of_limit_reference_is_recorded_never_clipped(self):
        q = np.full((151, 20), .98, dtype=np.float32)
        changed, _, _, _ = GENERATOR.corrupt_q_target(q, np.arange(151)*.02, .06, np.ones(20))
        self.assertGreater(float(changed[-1, 0]), 1.)
        excess, report = GENERATOR.limit_diagnostics(changed, np.full(20, -1.), np.ones(20), .001)
        self.assertGreater(report["strict_limit_violation_entries"], 0)
        self.assertFalse(report["clipped_during_generation"])
        self.assertGreater(float(excess[-1].max()), .039)

    def test_protocol_only_rebinds_npz_hash(self):
        parent = {"reference_npz_sha256": "base", "position_max_m": .02,
                  "progress_frame": {"mode": "same", "normal": [0, 0, 1]}}
        rebound, digest = GENERATOR.rebind_protocol(parent, "new")
        self.assertEqual(parent["reference_npz_sha256"], "base")
        self.assertEqual({key for key in parent if parent[key] != rebound[key]}, {"reference_npz_sha256"})
        self.assertEqual(digest, protocol_digest(rebound))


@unittest.skipUnless(BASE.is_file() and BUNDLE.is_dir(), "requires the local frozen pilot reference")
class FrozenVariantArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "variants"
        cls.result = GENERATOR.create_variants(BASE, cls.output, contact_bundle=BUNDLE)
        with np.load(BASE, allow_pickle=False) as raw:
            cls.base_reference = {key: raw[key].copy() for key in raw.files}
        cls.base_meta = json.loads(BASE.with_suffix(".json").read_text(encoding="utf-8"))
        cls.base_catalog = GENERATOR.resolve(cls.base_meta["provenance"]["source_file"])
        with np.load(cls.base_catalog, allow_pickle=False) as raw:
            cls.base_raw = {key: raw[key].copy() for key in raw.files}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_three_singleclip_catalogs_parse_and_hashes_verify(self):
        from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
        for variant in self.result["variants"]:
            resolved = resolve_recipe(variant["recipe_path"], clips_root=variant["catalog_root"], verify_clip_hashes=True)
            self.assertEqual(sum(len(group.clips) for group in resolved.clip_sets), 1)
            self.assertEqual(GENERATOR.digest(variant["catalog_clip"]), variant["catalog_clip_sha256"])

    def test_initial_derivatives_independent_targets_and_delta_exact(self):
        for variant in self.result["variants"]:
            folder = Path(variant["folder"])
            with np.load(variant["reference_file"], allow_pickle=False) as raw:
                ref = {key: raw[key].copy() for key in raw.files}
            with np.load(variant["catalog_clip"], allow_pickle=False) as raw:
                catalog = {key: raw[key].copy() for key in raw.files}
            with np.load(folder / "perturbation.npz", allow_pickle=False) as raw:
                delta = raw["applied_delta_rad"].copy()
            np.testing.assert_array_equal(delta, ref["joint_pos_rad"]-self.base_reference["joint_pos_rad"])
            for key in ref:
                if key != "joint_pos_rad":
                    np.testing.assert_array_equal(ref[key], self.base_reference[key])
            for key in catalog:
                if key != "reference_qpos":
                    np.testing.assert_array_equal(catalog[key], self.base_raw[key])
                    self.assertEqual(catalog[key].dtype, self.base_raw[key].dtype)
            np.testing.assert_array_equal(ref["joint_pos_rad"][:2], self.base_reference["joint_pos_rad"][:2])
            np.testing.assert_array_equal(np.diff(ref["joint_pos_rad"][:2], axis=0)/.02,
                                          np.diff(self.base_reference["joint_pos_rad"][:2], axis=0)/.02)
            self.assertTrue(all(variant["initial_conditions_and_target_preservation"].values()))

    def test_zero_control_copies_source_npz_bytes_and_protocol(self):
        zero = self.result["variants"][0]
        self.assertEqual(GENERATOR.digest(zero["reference_file"]), GENERATOR.digest(BASE))
        self.assertEqual(GENERATOR.digest(zero["catalog_clip"]), GENERATOR.digest(self.base_catalog))
        self.assertEqual(zero["protocol_sha256"], self.base_meta["task_acceptance_sha256"])

    def test_protocol_criteria_split_and_contact_targets_remain_base(self):
        for variant in self.result["variants"]:
            ref = Reference.load(variant["reference_file"])
            ref.validate()
            protocol = ref.meta["task_acceptance"]
            parent = self.base_meta["task_acceptance"]
            for key in parent:
                if key != "reference_npz_sha256":
                    self.assertEqual(protocol[key], parent[key])
            self.assertEqual(protocol["reference_npz_sha256"], GENERATOR.digest(variant["reference_file"]))
            self.assertEqual(ref.meta["task_acceptance_sha256"], protocol_digest(protocol))
            self.assertEqual(ref.meta["split_group"], self.base_meta["split_group"])
            self.assertIsNone(ref.meta["controlled_perturbation"]["original_video_parent_id"])
            contact = ref.meta["controlled_perturbation"]["contact_intent"]
            self.assertEqual(contact["reference_npz_sha256"], GENERATOR.digest(BASE))
            self.assertFalse(contact["recomputed_from_corrupted_q"])
            self.assertEqual(contact["target_npz_sha256"], GENERATOR.digest(BUNDLE / "contact_targets.npz"))

    def test_catalog_tamper_is_rejected_by_upstream_parser(self):
        from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
        source = self.result["variants"][2]
        with tempfile.TemporaryDirectory() as directory:
            copy_root = Path(directory) / "catalog"
            shutil.copytree(source["catalog_root"], copy_root)
            clip = copy_root / "controlled_q_bias" / Path(source["catalog_clip"]).name
            clip.write_bytes(clip.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                resolve_recipe(source["recipe_path"], clips_root=copy_root, verify_clip_hashes=True)


if __name__ == "__main__":
    unittest.main()
