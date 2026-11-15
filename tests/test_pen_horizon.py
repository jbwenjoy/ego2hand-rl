import importlib.util
import unittest
from pathlib import Path
import numpy as np
import torch
from supermimic_research.runtime.pen_native_planning_math import candidate_sequence_targets, candidate_targets, score_candidates, score_sequence


def fixture(H=5):
    cfg=dict(COM_scale_m=.005,shaft_scale_deg=5.,hand_scale_rad=.02,residual_scale_rad=.02,hand_weight=.01,residual_weight=.01,
        force_scale_N=.1,torque_scale_Nm=.005,max_penetration_m=.002,max_self_penetration_m=.0002,
        max_joint_limit_excursion_rad=.001,max_object_nonhand_normal_force_N=.1,orientation_objective='FULL_SO3')
    q=torch.zeros(2,H*4,27);q[:,:,3]=1
    args=(q,torch.zeros(2,H*4,26),torch.zeros(2,H*4,20),torch.zeros(2,H*4,6),torch.zeros(2,3),
        torch.zeros(2,H*4),torch.zeros(2,H*4,dtype=torch.int32),torch.zeros(2,H,20),torch.arange(7,27),0)
    ref=dict(COM=torch.zeros(H,3),shaft=torch.tensor([1.,0,0]).expand(H,-1),hand=torch.zeros(H,20),
        wrench=torch.zeros(H,6),valid=np.ones(H,bool),quaternion=torch.tensor([1.,0,0,0]).expand(H,-1))
    return args,ref,cfg


class HorizonTests(unittest.TestCase):
    def test_sequential_ema_reference_bias_clipping_and_unmodified_memory(self):
        raw=torch.zeros(2,5,20);raw[0,:,0]=1;raw[1,:,0]=-1
        initial=torch.zeros(1,20);ref=torch.arange(5)[:,None].expand(-1,20)*.01
        PD,ema,intended=candidate_sequence_targets(raw,initial,ref,torch.full((1,20),.02),torch.full((1,20),-.3),torch.full((1,20),.3))
        np.testing.assert_allclose(ema[0,:,0],[.25,.375,.4375,.46875,.484375])
        torch.testing.assert_close(intended[0,:,0],ref[:,0]+ema[0,:,0]-.02)
        self.assertEqual(float(PD[0,-1,0]),float(torch.tensor(.3)))
        self.assertTrue(torch.equal(initial,torch.zeros_like(initial)))
        first=candidate_targets(raw[:,0],initial,ref[0],torch.full((1,20),.02),torch.full((1,20),-.3),torch.full((1,20),.3))
        for a,b in zip((PD,ema,intended),first):self.assertTrue(torch.equal(a[:,0],b))
        with self.assertRaises(ValueError):candidate_sequence_targets(raw,initial,ref[:4],0,-1,1)

    def test_future_cost_changes_choice_and_later_safety_takes_precedence(self):
        args,ref,cfg=fixture();args[0][0,4:,0]=.02;args[0][1,:4,0]=.001
        self.assertEqual(int(score_sequence(*args,ref,0.,cfg)['index']),1)
        args[5][1,-1]=.2
        result=score_sequence(*args,ref,0.,cfg)
        self.assertFalse(bool(result['feasible'][1]));self.assertEqual(int(result['index']),0)
        args[6][0,0]=1
        result=score_sequence(*args,ref,0.,cfg)
        self.assertFalse(bool(result['feasible'].any()));self.assertEqual(int(result['index']),1)

    def test_per_control_wrench_mask_and_axial_rotation(self):
        args,ref,cfg=fixture(2);args[3][0,4:,2]=1
        result=score_sequence(*args,ref,1.,cfg)
        self.assertGreater(float(result['support'][0]),0)
        ref['valid'][1]=False
        result=score_sequence(*args,ref,1.,cfg)
        self.assertEqual(float(result['total'][0]),float(result['total'][1]))
        args[0][0,-1,3]=0;args[0][0,-1,4]=1
        result=score_sequence(*args,ref,0.,cfg)
        self.assertAlmostEqual(float(result['orientation_error_rad'][0]),np.pi,places=6)
        self.assertEqual(float(result['shaft_error_rad'][0]),0)
        self.assertEqual(int(result['index']),1)

    def test_H1_sequence_and_preserved_default_are_bitwise_equal(self):
        args,ref,cfg=fixture(1)
        old_path=Path(__file__).resolve().parents[1]/'runs/pen_horizon_legacy_sources_v1/snapshot/src/supermimic_research/runtime/pen_native_planning_math.py'
        # The portable H1 assertion stands independently of local ignored evidence.
        direct_args=(*args[:7],args[7][:,0],args[8],args[9],ref['COM'][0],ref['shaft'][0],ref['hand'][0],ref['wrench'][0],True,0.,cfg)
        direct=score_candidates(*direct_args,reference_quaternion=ref['quaternion'][0])
        sequence=score_sequence(*args,ref,0.,cfg)
        for key in direct:self.assertTrue(torch.equal(direct[key],sequence[key]),key)
        if old_path.exists():
            spec=importlib.util.spec_from_file_location('frozen_H1_math',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            original=old.score_candidates(*direct_args,reference_quaternion=ref['quaternion'][0])
            for key in original:self.assertTrue(torch.equal(direct[key],original[key]),key)

    def test_short_tail_uses_only_real_reference_controls(self):
        args,ref,cfg=fixture(3)
        result=score_sequence(*args,ref,0.,cfg)
        self.assertEqual(int(result['index']),0)
        ref['COM']=torch.zeros(5,3)
        with self.assertRaises(ValueError):score_sequence(*args,ref,0.,cfg)
