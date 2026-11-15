import copy
import tomllib
import unittest
from pathlib import Path
from supermimic_research.learning.pen_contrast_fit import common_model_config
from supermimic_research.learning.pen_response import pen_families


class CommonResponseFitConfig(unittest.TestCase):
    def config(self):return tomllib.loads((Path(__file__).resolve().parents[1]/'configs/pen_contact_response_fit_v1.toml').read_text())

    def test_data_schema2_explicitly_builds_fresh_model_schema1(self):
        study=self.config();original=copy.deepcopy(study);model=common_model_config(study)
        self.assertEqual(study,original);self.assertEqual(study['schema'],2)
        self.assertEqual(model['schema'],1);self.assertEqual(model['study_schema'],2)
        self.assertEqual(pen_families(model),('DIRECT','IMPULSE'))

    def test_CV_family_CHECK_selection_and_budget_reinterpretations_reject(self):
        for changes in [dict(model_comparison='DIRECT_VS_DIRECT_CV'),dict(CHECK_role='FINAL_TEST'),dict(planned_updates=100),dict(models=['POINT_DIRECT','POINT_DIRECT_CV'])]:
            study={**self.config(),**changes}
            with self.assertRaises(ValueError):common_model_config(study)
