import unittest
import numpy as np
from supermimic_research.mechanics.pen_interval_targets import interval_targets,TRACE_KEYS


class IntervalTargetsTest(unittest.TestCase):
    def fixture(self):
        cfg=dict(object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001,nonhand_normal_N=.1,geometry_soft_weight=.25,geometry_failure_penalty=2.)
        trace={k:np.zeros(8) for k in TRACE_KEYS};trace[TRACE_KEYS[0]][0]=.001
        raw=np.zeros((3,24));raw[:,:3]=[[1,0,2],[1,0,2],[2,0,1]];raw[:,18]=[.03,.04,50.];trace['raw_contact_tick_plus23']=raw
        path=np.array([.001,0,0,.07]);loss=np.float32(np.log1p((path/np.array([.002,.0002,.001,.1]))**2).sum())
        window=dict(geometry_max=path[:3],nonhand_max_sum=path[3:4],counts=np.array([8,3]),errors=np.array(0),global_error=np.array([0]),objective_geometry_loss=loss,objective_geometry_failure=False)
        return trace,window,[1,2,0],['object','robot/right_finger1_link1','floor'],1,cfg

    def test_interval_max_and_nonhand_sum_ignore_robot_normal(self):
        result=interval_targets(*self.fixture(),phase_valid=True)
        self.assertTrue(all(result['checks'].values()));np.testing.assert_array_equal(result['path4'],[.001,0,0,.07])
        self.assertAlmostEqual(float(result['geometry_penalty']),-.25*float(result['geometry_loss']),places=6)

    def test_endpoint_substitution_rejected_by_comparison(self):
        args=list(self.fixture());args[1]['geometry_max']=np.zeros(3)
        result=interval_targets(*args,phase_valid=True)
        self.assertFalse(result['checks']['trace_geometry_matches_window'])

    def test_error_and_phase_failures_do_not_disappear(self):
        for defect in ['errors','phase']:
            args=list(self.fixture());args[1]['objective_geometry_failure']=True
            if defect=='errors':args[1]['errors']=np.array(128)
            result=interval_targets(*args,phase_valid=defect!='phase')
            self.assertTrue(result['geometry_failure']);self.assertTrue(result['checks']['original_failure'])
            self.assertAlmostEqual(float(result['geometry_penalty']),-.25*float(result['geometry_loss'])-2,places=6)

    def test_wrong_tick_count_rejected(self):
        args=list(self.fixture());args[0][TRACE_KEYS[0]]=np.zeros(7)
        with self.assertRaises(ValueError):interval_targets(*args,phase_valid=True)


if __name__=='__main__':unittest.main()
