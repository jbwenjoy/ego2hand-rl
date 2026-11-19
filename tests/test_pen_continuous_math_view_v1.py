import copy
import unittest
import numpy as np
import torch
from supermimic_research.runtime.pen_continuous_math_view_v1 import numeric_view,context,terminal_reason,physics_view,policy
from supermimic_research.mechanics.pen_reward_chain import A,I,ACTION,MOTION,ROBOT,CACHE,source_context
from supermimic_research.runtime.shared_snapshot_transport import validate


class ContinuousViewTests(unittest.TestCase):
    def fixture(self):
        t={ACTION+(A('_target_ids'),):torch.arange(20),ACTION+(A('_position_ctrl_ids'),):torch.arange(20),ACTION+(A('_position_ctrl_limited'),):torch.ones(20,dtype=torch.bool),
           ACTION+(A('_processed_actions'),):torch.zeros(1,20),ACTION+(A('_raw_actions'),):torch.zeros(1,20),MOTION+(A('time_steps'),):torch.tensor([1]),MOTION+(A('trajectory_ids'),):torch.tensor([0]),MOTION+(A('_trajectory_starts'),):torch.tensor([0]),
           MOTION+(A('group'),A('joint_pos')):torch.zeros(369,20),ROBOT+(A('joint_pos_limits'),):torch.tensor([[[-2.,2.]]*20]),ROBOT+(A('encoder_bias'),):torch.zeros(1,20),CACHE+(A('world_pose'),):torch.zeros(1,369,7),
           (A('support_table'),A('targets')):torch.zeros(1,368,6,dtype=torch.float64),(A('support_table'),A('valid')):torch.ones(368,dtype=torch.bool)}
        cfg=(A('env'),A('cfg'),A('actions'),I('joint_pos'))
        s={ACTION+(A('_scale'),):.5,cfg+(A('alpha'),):.5,cfg+(A('residual_clip'),I(0)):-.6,cfg+(A('residual_clip'),I(1)):.6,ACTION+(A('_mask_enabled'),):False,(A('support_weight'),):.2}
        initial=dict(schema='controlled_pen_owned_state:4',worlds=1,metadata=dict(fixed_frame_features='LIVE'),python=dict(tensors=t,scalars=s),native=dict(model=dict(arrays={'actuator_ctrllimited':np.ones(20,bool),'actuator_ctrlrange':np.tile([-2.,2.],(1,20,1))}),data={}))
        dynamic=dict(phase=10,EMA=torch.ones(1,20)*.1,last_raw=torch.zeros(1,20),trajectory_ids=torch.tensor([0]),mount=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]),cached_R=np.eye(3),arrays={'qpos':np.zeros((1,27),np.float32),'qvel':np.zeros((1,26),np.float32),'qacc_warmstart':np.zeros((1,26),np.float32),'time':np.array([.18],np.float32)})
        return initial,dynamic,dict(aids=list(range(20)))

    def test_reduced_context_matches_full_numeric_values_and_is_not_a_checkpoint(self):
        initial,dynamic,case=self.fixture(); expected=copy.deepcopy(initial); expected['python']['tensors'][ACTION+(A('_processed_actions'),)]=dynamic['EMA']; expected['python']['tensors'][MOTION+(A('time_steps'),)]=torch.tensor([10])
        a=context(initial,dynamic,case); b=source_context(dict(native=expected,phase=10),case)
        for k,v in a.items(): self.assertTrue(torch.equal(v,b[k]) if isinstance(v,torch.Tensor) else v==b[k])
        view=numeric_view(initial,dynamic)
        with self.assertRaises(ValueError): validate(view)
        view['native']['data']['arrays']['qpos'][:]=9
        self.assertEqual(float(dynamic['arrays']['qpos'].sum()),0)
        self.assertTrue(torch.equal(initial['python']['tensors'][MOTION+(A('time_steps'),)],torch.tensor([1])))

    def test_invalid_layout_and_failures_remain_censored(self):
        initial,dynamic,_=self.fixture(); dynamic['phase']=0
        with self.assertRaises(ValueError): numeric_view(initial,dynamic)
        self.assertEqual(terminal_reason(False,False,False,2),'PHYSICAL_AUDIT_STOP')
        self.assertEqual(terminal_reason(False,True,False,2),'PHYSICAL_AUDIT_STOP')
        self.assertEqual(terminal_reason(True,True,False,2),'TASK_FAILURE')
        self.assertEqual(terminal_reason(True,False,True,368),'REFERENCE_COMPLETION')
        self.assertIsNone(terminal_reason(True,False,False,20))

    def test_policy_and_physical_axes_use_declared_nonzero_object_address(self):
        initial,dynamic,case=self.fixture(); t=initial['python']['tensors']
        t[MOTION+(A('group'),A('link_pos'))]=torch.zeros(369,21,3)
        t[ROBOT+(A('default_joint_pos'),)]=torch.zeros(1,20); t[ROBOT+(A('default_joint_vel'),)]=torch.zeros(1,20)
        t[CACHE+(A('world_pose'),)][:,:,3]=1.
        dynamic['arrays']['qpos'][0,23]=1.
        case.update(qids=list(range(20)),dofs=list(range(20)),model_layout=dict(object_qposadr=20))
        expected=torch.zeros(1,161); expected[0,23]=1.; expected[0,27]=1.; expected[0,144]=1.; expected[0,148]=1.
        # Layout: reference q20,pose9,links63,q20,qd20,current pose9,RAW20.
        expected.zero_(); expected[0,[23,27,135,139]]=1.
        self.assertTrue(torch.equal(policy(initial,dynamic,case),expected))
        np.testing.assert_array_equal(physics_view(initial,dynamic,case)['source_qpos_R'],np.eye(3))


if __name__=='__main__': unittest.main()
