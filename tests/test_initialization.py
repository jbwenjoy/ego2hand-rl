"""Mechanics checks against analytic shapes and the actual rigid plant."""
from pathlib import Path
from itertools import product
import json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from supermimic_research.mechanics.convex import ConvexPolytope
from supermimic_research.mechanics.collision import segment_distance,CollisionAudit
from supermimic_research.mechanics.initialization import NativeInitialization
from supermimic_research.artifacts import ArtifactRegistry
from plan_com_relocation import minimum_relocation,maximum_force_ignoring_object_moment


class InitializationTests(unittest.TestCase):
    def test_convex_translation_distance_and_rigid_frame_invariance(self):
        from scipy.spatial.transform import Rotation
        cube=np.array(list(product((-1.,1.),repeat=3)))
        far=cube+np.array([3.,0.,0.]);inside=cube+np.array([.2,0.,0.])
        self.assertAlmostEqual(ConvexPolytope(cube).convex(far)['signed_translation_margin_m'],1.)
        self.assertAlmostEqual(ConvexPolytope(cube).convex(inside)['signed_translation_margin_m'],-1.8)
        R=Rotation.from_rotvec([.2,.3,-.5]).as_matrix();t=np.array([.4,-.7,.2])
        self.assertAlmostEqual(ConvexPolytope(cube@R.T+t).convex(far@R.T+t)['signed_translation_margin_m'],1.)
        capsule=ConvexPolytope(cube).capsule(np.array([3.,-.5,0.]),np.array([3.,.5,0.]),.25)
        self.assertAlmostEqual(capsule['signed_translation_margin_m'],1.75)

    def test_segment_parallel_crossing_and_point_degeneracy(self):
        z=np.zeros(3);one=np.array([1.,0.,0.]);up=np.array([0.,1.,0.])
        self.assertAlmostEqual(segment_distance(z,one,up,one+up),1.)
        self.assertEqual(segment_distance(z,one,np.array([.5,-1.,0.]),np.array([.5,1.,0.])),0.)
        self.assertAlmostEqual(segment_distance(z,z,2*one,3*one),2.)

    def test_relaxed_COM_plan_has_correct_moment_sign_and_trust_bound(self):
        W=np.array([[0.],[0.],[1.],[0.],[-.02],[0.]])
        loads=np.zeros((20,1));z=np.zeros(20)
        answer=minimum_relocation(W,loads,z,-np.ones(20),np.ones(20),z)
        self.assertTrue(answer['success'])
        self.assertAlmostEqual(answer['witness']['delta_COM_world_m'][0],.02)
        self.assertAlmostEqual(answer['witness']['horizontal_Linf_m'],.02)
        self.assertFalse(minimum_relocation(W,loads,z,-np.ones(20),np.ones(20),z,trust=.01)['success'])
        loads[0,0]=1
        force_only=maximum_force_ignoring_object_moment(W,loads,z,-np.full(20,2.),np.full(20,2.),z)
        self.assertTrue(force_only['success']);self.assertAlmostEqual(force_only['witness']['maximum_vertical_force_N'],2.)

    def test_actual_plant_net_force_mass_relation_and_original_decode(self):
        import mujoco
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml');bundle=json.loads(registry.resolve('phone_task').read_text(encoding='utf8'))
        model=mujoco.MjModel.from_binary_path(str(registry.resolve('phone_model')));model.opt.timestep=.0003125
        q=np.array(bundle['initial']['qpos27_F64']);u=np.array(bundle['initial']['ctrl20_F64']);engine=NativeInitialization(model,q,u)
        q1,u1=engine.decode(np.zeros(38));self.assertEqual(q1.tobytes(),q.tobytes());self.assertEqual(u1.tobytes(),u.tobytes())
        inertia_acc=np.empty(26);mujoco.mj_mulM(model,engine.data,inertia_acc,engine.data.qacc)
        np.testing.assert_allclose(inertia_acc,engine.base['net_generalized26'],rtol=1e-10,atol=1e-10)
        audit=CollisionAudit(model);self.assertEqual(len(audit.hand)+1,22);self.assertEqual(len(audit.self),189)
        self.assertEqual(engine.counts['integration_steps'],0)
        bad=np.zeros(38);bad[0]=2
        with self.assertRaises(ValueError):engine.decode(bad)
        proposed=np.zeros(38);proposed[12]=.25;proposed[15]=.1
        moved,_=engine.decode(proposed);deviation=engine.deviation(moved)
        np.testing.assert_allclose(np.array(deviation['proposed_reference_left_transform_R'])@q[20:23]+deviation['proposed_reference_left_transform_t_m'],moved[20:23],atol=1e-14)
        self.assertFalse(deviation['Actor_or_training_compatible'])


if __name__=='__main__':unittest.main()
