"""Control-map, global-baseline and approximate force-balance invariants."""
from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from control_authority import schedules,ScheduleController,compare_trace,static_force_balance
from contact_torque_box import pd_bounds,solve_box
from review_control_authority import native_pyramid_rays,support_interval
from supermimic_research.artifacts import ArtifactRegistry,file_pin
from supermimic_research.learning import runner as R


class AuthorityTests(unittest.TestCase):
    def test_predeclared_complete65_plan_and_no_action_clamp(self):
        a,b=schedules(),schedules()
        self.assertEqual(len(a),65);self.assertEqual(len({name for name,_ in a}),65)
        self.assertTrue(np.array_equal(a[0][1],np.zeros((100,20))))
        for (name,x),(other,y) in zip(a,b):
            self.assertEqual(name,other);self.assertEqual(x.dtype,np.float64)
            self.assertEqual(x.tobytes(),y.tobytes());self.assertLessEqual(abs(x).max(),1.)
        for axis in range(20):
            for index,sign in [(1+2*axis,-1),(2+2*axis,1)]:
                self.assertTrue((a[index][1][:,axis]==sign).all())
                self.assertTrue((a[index][1][:,np.arange(20)!=axis]==0).all())
        self.assertEqual(65*(6400+12800),1248000)

    def test_real_residual_EMA_maps_half_then_three_quarters_scale(self):
        import torch
        path,c,registry=R.configuration(ROOT/'configs/phone_online_gpu.toml')
        task=R._task(registry,c,'A1_tracking',None)
        cb=ScheduleController(task,np.ones((100,20)),.0003125)
        base=np.array(task.b['initial']['ctrl20_F64']);scale=np.array(task.b['action']['symmetric_residual_scale20_rad'])
        first=cb(0,.0003125,None,None);second=cb(64,.0003125,None,None)
        np.testing.assert_array_equal(first,base+.5*scale);np.testing.assert_array_equal(second,base+.75*scale)
        self.assertEqual(task.flatten_memory(cb.memory).dtype,torch.float64)
        self.assertEqual(cb.records[0]['memory83_before'][0],0);self.assertEqual(cb.records[1]['memory83_before'][0],1)
        with self.assertRaises(ValueError):cb(64,.0003125,None,None)

    def test_raw_baseline_barrier_rejects_signed_zero_difference(self):
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml')
        with tempfile.TemporaryDirectory() as tmp:
            one=Path(tmp)/'one.npz';two=Path(tmp)/'two.npz'
            np.savez(one,value=np.array([0.,np.nan]));np.savez(two,value=np.array([0.,np.nan]))
            self.assertTrue(all(compare_trace(file_pin(one),two,registry).values()))
            np.savez(two,value=np.array([-0.,np.nan]))
            self.assertFalse(all(compare_trace(file_pin(one),two,registry).values()))

    def test_static_LP_only_balances_when_contact_directions_allow_it(self):
        rows=np.zeros((2,24));rows[:,1]=[15,28];rows[:,2]=71;rows[:,3]=3
        rows[:,6]=[-.1,.1];frame=np.array([[0,0,1],[1,0,0],[0,1,0]],float);rows[:,9:18]=frame.reshape(9)
        result=static_force_balance(rows,np.zeros(3),mu=0.,rays=4)
        self.assertTrue(result['LP_success']);self.assertAlmostEqual(result['minimum_sum_normal_coefficients_N'],1.962)
        self.assertLess(np.linalg.norm(result['force_residual_world_N']),1e-9)
        self.assertLess(np.linalg.norm(result['moment_residual_world_Nm']),1e-9)
        rows[:,9:12]*=-1
        impossible=static_force_balance(rows,np.zeros(3),mu=0.,rays=4)
        self.assertFalse(impossible['LP_success']);self.assertIsNone(impossible['force_residual_world_N'])

    def test_affine_PD_box_includes_actual_plant_force_saturations(self):
        low,high=pd_bounds(np.full(20,2.),np.zeros(20),np.full(20,-.05),np.full(20,.05),np.tile([-1.,1.],(20,1)),np.tile([-.08,.08],(20,1)))
        np.testing.assert_array_equal(low,np.full(20,-.08));np.testing.assert_array_equal(high,np.full(20,.08))

    def test_object_force_balance_is_not_sufficient_for_torque_box(self):
        wrench=np.zeros((6,1));wrench[2,0]=1;loads=np.zeros((20,1));loads[0,0]=1
        target=np.array([0.,0.,1.962,0.,0.,0.]);offset=np.zeros(20)
        narrow=solve_box(wrench,loads,target,offset,np.full(20,-.1),np.full(20,.1),.1)
        broad=solve_box(wrench,loads,target,offset,np.full(20,-5.),np.full(20,5.),.1)
        self.assertFalse(narrow['success']);self.assertIsNone(narrow['required_joint_torque_Nm'])
        self.assertTrue(broad['success']);self.assertAlmostEqual(broad['required_joint_torque_Nm'][0],1.962)

    def test_native_condim3_pyramid_nonnegative_generators(self):
        rays=native_pyramid_rays(np.array([.7,.7]))
        self.assertEqual(rays.shape,(4,3));self.assertTrue((rays[:,0]>0).all())
        np.testing.assert_allclose(rays[:,0],np.ones(4),rtol=0,atol=0)
        np.testing.assert_allclose(rays[:,1:].sum(0),np.zeros(2),rtol=0,atol=0)
        self.assertTrue((abs(rays[:,1:])<=.7).all())
        np.testing.assert_array_equal(rays,np.array([[1,.7,0],[1,-.7,0],[1,0,.7],[1,0,-.7]]))

    def test_supported_weight_uses_joint_load_and_explicit_friction_envelope(self):
        wrench=np.zeros((6,1));wrench[2,0]=1
        loads=np.zeros((20,1));loads[0,0]=1
        interval=support_interval(wrench,loads,np.zeros(20),np.full(20,-1.),np.full(20,1.),np.full(20,.3),.1)
        self.assertAlmostEqual(interval['minimum']['support_weight_N'],0.)
        self.assertAlmostEqual(interval['maximum']['support_weight_N'],1.3)
        np.testing.assert_allclose(interval['maximum']['witness']['force_residual_world_N'],np.zeros(3),atol=1e-12)
        # Downward-only force cannot support positive weight even with ample torque.
        wrench[2,0]=-1
        downward=support_interval(wrench,loads,np.zeros(20),np.full(20,-1.),np.full(20,1.),np.full(20,.3),.1)
        self.assertAlmostEqual(downward['maximum']['support_weight_N'],0.)


if __name__=='__main__':unittest.main()
