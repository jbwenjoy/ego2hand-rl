"""Known poses/visibility are intake descriptors, not inferred contact truth."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.ego.clip_inventory import describe_window,pose


class EgoInventoryTests(unittest.TestCase):
    def rows(self,motion=.02):
        return [dict(frame=i,timestamp_ns=1000000000+round(i*1e9/30),objects={'32:marker':dict(id=32,uid='marker',name='whiteboard_marker',pose=dict(translation_xyz=[.1+motion*i/60,0.,0.],quaternion_wxyz=[1.,0.,0.,0.]),rgb_visibility=1.)},
            hands={'right':dict(pose=dict(translation_xyz=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.]),rgb_visibility=1.)}) for i in range(61)]

    def test_stationary_visible_object_is_not_a_moving_candidate(self):
        moving=describe_window(self.rows(),'32:marker','right');static=describe_window(self.rows(0.),'32:marker','right')
        self.assertTrue(moving['intake_eligible']);self.assertAlmostEqual(moving['elapsed_s'],2.)
        self.assertAlmostEqual(moving['object_world_max_translation_m'],.02);self.assertFalse(static['intake_eligible'])

    def test_missing_visibility_and_pose_are_not_filled_as_known(self):
        rows=self.rows()
        for row in rows:row['objects']['32:marker']['rgb_visibility']=None
        descriptor=describe_window(rows,'32:marker','right')
        self.assertEqual(descriptor['object_visibility_known_frames'],0);self.assertFalse(descriptor['intake_eligible'])
        rows[5]['hands']['right']['pose']=None
        self.assertIsNone(describe_window(rows,'32:marker','right'));self.assertIsNone(pose(None))


if __name__=='__main__':unittest.main()
