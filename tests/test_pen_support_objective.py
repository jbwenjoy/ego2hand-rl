"""Timing/origin/gate checks before the registered geometry/support pilot."""
import unittest
import numpy as np
import torch
from supermimic_research.mechanics.pen_support_objective import objective_terms,select_interval
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,SCHEMA
from supermimic_research.learning.owned_pen_ppo import OwnedPenPPO


CFG=dict(object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001,nonhand_normal_N=.1,force_scale_N=.3,torque_scale_Nm=.003,geometry_soft_weight=.25,geometry_failure_penalty=2.)


def fixture():
    R=torch.eye(3).repeat(2,1,1);R[1]=torch.tensor([[0.,0,1],[0,1,0],[-1,0,0]])
    world=torch.tensor([[0.,0,.3,0,.006,0]]*2,dtype=torch.float64)
    target=torch.cat(((R.double().transpose(-1,-2)@world[:,:3,None]).squeeze(-1),(R.double().transpose(-1,-2)@world[:,3:,None]).squeeze(-1)),-1)
    wrench=torch.zeros(2,8,6,dtype=torch.float64);wrench[:,5]=target*8;wrench[:,7]=999
    return dict(tracking=torch.ones(2),native_term=torch.zeros(2,dtype=torch.bool),geometry=torch.zeros(2,3,dtype=torch.float64),nonhand=torch.zeros(2,2,dtype=torch.float64),
        errors=torch.zeros(2,dtype=torch.int32),global_error=torch.zeros(1,dtype=torch.int32),wrench_sum=wrench,counts=torch.tensor([[8,2],[8,4]],dtype=torch.int32),source_R=R,
        target_world=world,target_valid=torch.ones(2,dtype=torch.bool),phase_valid=torch.ones(2,dtype=torch.bool),cfg=CFG,weight=.2)


class SupportObjectiveTests(unittest.TestCase):
    def test_source_axes_and_moving_COM_slot5_ignore_fixed_origin_slot7(self):
        a=fixture();r=objective_terms(**a)
        torch.testing.assert_close(r['target_source'][1,:3],torch.tensor([-.3,0,0],dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(r['support_loss'],torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(r['reward'],torch.full((2,),1.2),atol=1e-7,rtol=0)

    def test_original_depth_gate_disables_bonus_without_repairing_geometry(self):
        a=fixture();a['geometry'][0,0]=.002001;before=a['geometry'].clone();r=objective_terms(**a)
        self.assertTrue(r['geometry_failure'][0]);self.assertFalse(r['support_eligible'][0]);self.assertEqual(float(r['support_bonus'][0]),0)
        self.assertTrue(torch.equal(before,a['geometry']));self.assertLess(float(r['reward'][0]),-1)

    def test_native_failure_nonhand_and_self_gates_cannot_receive_support(self):
        for field in ['native','nonhand','self','error']:
            a=fixture()
            if field=='native':a['native_term'][0]=True
            if field=='nonhand':a['nonhand'][0,0]=.10001
            if field=='self':a['geometry'][0,1]=.000201
            if field=='error':a['global_error'][0]=1
            r=objective_terms(**a);self.assertEqual(float(r['support_bonus'][0]),0)

    def test_invalid_derivative_tail_has_no_bonus_but_is_not_geometry_failure(self):
        a=fixture();a['target_valid'][0]=False;r=objective_terms(**a)
        self.assertFalse(r['geometry_failure'][0]);self.assertEqual(float(r['support_bonus'][0]),0);self.assertEqual(float(r['reward'][0]),1)

    def test_used_phase_one_selects_interval_zero_and_bad_phase_is_explicit(self):
        targets=torch.arange(2*4*6,dtype=torch.float64).reshape(2,4,6);valid=torch.tensor([True,False,False,False])
        value,mask,phase_valid=select_interval(targets,valid,torch.tensor([1,4]))
        torch.testing.assert_close(value[0],targets[0,0]);torch.testing.assert_close(value[1],targets[1,3]);self.assertEqual(mask.tolist(),[True,False]);self.assertTrue(phase_valid.all())
        _,mask,phase_valid=select_interval(targets,valid,torch.tensor([0,5]));self.assertFalse(mask.any());self.assertFalse(phase_valid.any())

    def test_reference_mean_constant_acceleration_and_three_invalid_tail_intervals(self):
        t=np.arange(8)*.02;a=np.array([.2,-.1,.3]);p=.5*t[:,None]**2*a;q=np.tile([1.,0,0,0],(8,1));g=np.array([0,0,-9.81])
        r=mean_reference_wrench(p,q,.03,np.diag([.001,.002,.003]),g,.02)
        np.testing.assert_allclose(r['wrench_world'][:4,:3],np.tile(.03*(a-g),(4,1)),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(r['valid_interval'],[True]*4+[False]*3)

    def test_public_training3_and_training2_schemas_reject_cross_loading(self):
        three=SupportPenPPO.__new__(SupportPenPPO);two=OwnedPenPPO.__new__(OwnedPenPPO)
        with self.assertRaisesRegex(ValueError,'training3'):three.preflight({'schema':'pen_owned_contact_PPO_training:2'})
        with self.assertRaisesRegex(ValueError,'training2'):two.preflight({'schema':SCHEMA})


if __name__=='__main__':unittest.main()
