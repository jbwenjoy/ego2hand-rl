from pathlib import Path
import sys,json,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts


class WarpSavedReviewTests(unittest.TestCase):
    def test_duplicate_variant_dt_is_rejected_before_wrench_artifacts_can_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'report.json';output=root/'review'
            report.write_text(json.dumps(dict(sources=[],cases=[
                dict(name='ORIGINAL',controller='CONSTANT_RESIDUAL',dt=.005),
                dict(name='ORIGINAL',controller='CONSTANT_RESIDUAL',dt=.005)])),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'unique expected cohort'):
                review_saved_warp_contacts(report,output,{('ORIGINAL',.005),('PROJECTED',.005)})
            self.assertFalse(list(output.glob('*.npz')))


if __name__=='__main__':unittest.main()
