"""Native Jacobian/bounds and compiled eligibility independent checks."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.mechanics.retargeting import HandTipIK
from supermimic_research.mechanics.collision import self_pairs
from supermimic_research.mechanics.compound_collision import eligible_pairs


class RetargetingTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.mj=mujoco; self.model=mujoco.MjModel.from_binary_path(str(ArtifactRegistry(ROOT/'configs/artifacts.toml').resolve('phone_model')))
        self.state=mujoco.MjData(self.model);mujoco.mj_kinematics(self.model,self.state);mujoco.mj_comPos(self.model,self.state)
        self.targets=self.state.site_xpos[[self.model.site(f'right_finger{i}_tip').id for i in range(1,6)]].copy()

    def test_analytic_native_tip_jacobian_matches_central_difference(self):
        engine=HandTipIK(self.model,self.state.qpos,self.targets)
        q=np.full(20,.1);analytic=engine.jacobian(q)
        numeric=[]
        for i in range(20):
            delta=np.zeros(20);delta[i]=1e-6
            numeric.append((engine.residual(q+delta)-engine.residual(q-delta))/(2e-6))
        np.testing.assert_allclose(analytic,np.array(numeric).T,atol=1e-8,rtol=1e-7)
        before=engine.counts['kinematics'];engine.state(q);engine.state(q)
        self.assertLessEqual(engine.counts['kinematics']-before,1)

    def test_solver_positions_do_not_imply_grasp_and_bounds_are_not_clamped(self):
        engine=HandTipIK(self.model,self.state.qpos,self.targets)
        result=engine.solve(np.full(20,.1),max_nfev=100)
        self.assertLess(result['tip_errors_m'].max(),1e-5)
        self.assertFalse(result['source_tip_alignment_is_contact_qualification'])
        with self.assertRaises(ValueError):engine.state(engine.hi+1)
        self.assertEqual(engine.counts['integration_steps'],0)

    def test_generic_compiled_pair_filter_reproduces_frozen_hand_set(self):
        hand,expected=self_pairs(self.model)
        self.assertEqual(expected,eligible_pairs(self.model,hand));self.assertEqual(len(expected),189)


if __name__=='__main__':unittest.main()
