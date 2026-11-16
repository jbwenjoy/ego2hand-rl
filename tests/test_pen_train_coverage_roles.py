"""CHECK packets must never acquire TRAIN eligibility through preparation."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from supermimic_research.artifacts import file_pin

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('train_coverage_preparation',ROOT/'tools/prepare_pen_train_coverage.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PreparationRoles(unittest.TestCase):
    def rejects(self,**changes):
        run=dict(schema='trained_pen_same_source_response_probe:2',config=file_pin(ROOT/'configs/pen_response_train_coverage_v1.toml'),role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE',restore_mode='STRICT',CHECK_response_rows_used=0,passed=5,total=5)
        run.update(changes)
        with tempfile.TemporaryDirectory() as td:
            report=Path(td)/'report.json';report.write_text(json.dumps(run),encoding='utf-8')
            out=ROOT/'runs'/('MUST_NOT_CREATE_'+Path(td).name)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError,'CHECK cannot be relabelled'):module.main(SimpleNamespace(report=report,output=out))
            self.assertFalse(out.exists())

    def test_CHECK_role_rejects_before_any_output(self):
        self.rejects(role='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT')

    def test_CHECK_rows_cannot_be_mixed_into_TRAIN(self):
        self.rejects(CHECK_response_rows_used=1)

    def test_experimental_epoch_cannot_impersonate_strict_cohort(self):
        self.rejects(restore_mode='READ_LOCKED_EPOCH')

    def test_reserved_fullrank_validation_cannot_be_prepared_as_TRAIN(self):
        self.rejects(schema='trained_pen_same_source_response_probe:3',config=file_pin(ROOT/'configs/pen_response_fullrank_validation_v1.toml'))

    def test_fullrank_TRAIN_cannot_be_mislabelled_as_reserved_validation(self):
        self.rejects(schema='trained_pen_same_source_response_probe:3',config=file_pin(ROOT/'configs/pen_response_fullrank_train_v1.toml'),role='VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT')
