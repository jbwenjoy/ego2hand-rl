"""Saved-only heldout-loader and zero/unknown response semantics."""
from pathlib import Path
import copy,json,sys,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from supermimic_research.artifacts import ArtifactRegistry
from analyze_online_response import load_probe,response_stats


class ResponseTests(unittest.TestCase):
    def test_saved_all_contexts_axes_masks_and_forbidden_training_role(self):
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml')
        report=json.loads(registry.resolve('action_response').read_text('utf8'));d=load_probe(report,registry)
        self.assertEqual((len(d['rows']),len(d['sides']),len(d['baseline_rows'])),(328,320,8))
        self.assertTrue(d['available'].all());self.assertTrue(d['known'].all())
        changed=copy.deepcopy(report);changed['permanent_training_eligible']=True
        with self.assertRaisesRegex(ValueError,'heldout'):load_probe(changed,registry)

    def test_exact_zero_prior_and_unknown_rows_do_not_become_zero_error(self):
        native=np.zeros((3,52));prediction=np.zeros((3,52));prediction[0,40:43]=[1,2,2];native[2]=np.nan;prediction[2]=np.nan
        result=response_stats(native,prediction,np.array([True,True,False]),np.array([0,1,1]))
        self.assertEqual(result[0]['rows'],2)
        self.assertEqual(result[0]['blocks']['COM_m']['zero_response_prior_RMS'],0.)
        self.assertIsNone(result[0]['blocks']['COM_m']['error_over_zero_prior'])
        self.assertAlmostEqual(result[0]['blocks']['COM_m']['component_error_RMS'],np.sqrt(1.5))
        self.assertEqual(len(result),21)
        self.assertEqual(result[3]['rows'],0)
        self.assertIsNone(result[3]['blocks']['COM_m']['component_error_RMS'])


if __name__=='__main__':unittest.main()
