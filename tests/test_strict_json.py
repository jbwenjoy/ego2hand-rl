import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from supermimic_research.strict_json import write_json


class StrictJsonTests(unittest.TestCase):
    def test_case_and_failure_records_accept_original_numpy_boolean_comparison(self):
        score=np.float64(2.);flag=abs(float(score)-score)<=2e-6
        case=dict(quality=dict(source_reward=flag),native_integrations=np.int64(8),score=np.float32(2.))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'failure.json';write_json(p,dict(type='OriginalError',cases=[case],attempted_native_integrations=8))
            record=json.loads(p.read_text());self.assertIs(record['cases'][0]['quality']['source_reward'],True);self.assertIs(type(record['cases'][0]['native_integrations']),int)
            self.assertEqual(record['attempted_native_integrations'],8)

    def test_array_requires_explicit_schema_and_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json'
            with self.assertRaises(TypeError):write_json(p,dict(unexpected=np.ones(4)))
            self.assertFalse(p.exists())
            with self.assertRaises(ValueError):write_json(p,dict(score=np.float32(np.nan)))
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main()
