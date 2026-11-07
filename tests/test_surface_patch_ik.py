"""Material-point kinematics preserve normals and have native Jacobian agreement."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.surface_patch_ik import SurfacePatchIK,material_anchor


class SurfacePatchTests(unittest.TestCase):
    def test_point_and_normal_jacobian_match_central_kinematics(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint axis="0 0 1"/><geom type="capsule" size=".01 .02" pos=".04 0 0"/></body></worldbody></mujoco>')
        anchor=dict(geom=0,point_local=np.array([.01,0,.01]),normal_local=np.array([1.,0,0]))
        problem=SurfacePatchIK(model,[0.],[0],[0],anchor,[[.05,.01,.01]],[[1.,0,0]],[17])
        x=np.array([.2]);jac=problem.evaluate(x)[1];h=1e-6
        secant=(problem.evaluate(x+h)[0]-problem.evaluate(x-h)[0])/(2*h)
        np.testing.assert_allclose(jac[:,0],secant,atol=2e-9,rtol=1e-8)

    def test_positive_standoff_is_outside_and_not_contact_truth(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide" axis="1 0 0"/><geom type="capsule" size=".01 .02"/></body></worldbody></mujoco>')
        anchor=dict(geom=0,point_local=np.array([.01,0,0]),normal_local=np.array([-1.,0,0]))
        problem=SurfacePatchIK(model,[0.],[0],[0],anchor,[[0,0,0]],[[-1.,0,0]],[4],stand_off_m=.001)
        result=problem.solve([0.],[-.02],[.02]);self.assertAlmostEqual(result['point_world'][0],.001,places=7)
        self.assertFalse(result['native_contact_or_force_certified'])

    def test_capsule_anchor_is_on_rounded_surface(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="capsule" size=".01 .02"/></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        point=np.array([1,0,1])/np.sqrt(2)*.01+[0,0,.02]
        anchor=material_anchor(model,data,0,point,[1,0,1])
        np.testing.assert_allclose(anchor['point_local'],point,atol=1e-14)
        np.testing.assert_allclose(anchor['normal_local'],np.array([1,0,1])/np.sqrt(2),atol=1e-14)


if __name__=='__main__':unittest.main()
