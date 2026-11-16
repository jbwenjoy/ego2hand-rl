import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_pen_reference_cache import fixture
from supermimic_research.runtime.owned_pen_runtime import OwnedPenFeatures,OwnedPenRuntime
from supermimic_research.runtime.pen_native_policy import NativePenFeatures
from supermimic_research.runtime.pen_reference_cache import FrozenPenReferenceCache
from supermimic_research.learning.pen_support_ppo import fork_fixed_frame_mode,CENTERED_SCHEMA
from supermimic_research.runtime import owned_pen_state as native


def feature_fixture():
    f=fixture();f.reference=f.motion._frames;f.action._processed_actions=torch.arange(40).reshape(2,20)*.003
    f.env.sim.mj_model=SimpleNamespace(jnt_dofadr=np.array([0]),body_jntadr=np.array([-1,0]))
    f.reference_cache=FrozenPenReferenceCache(f);f.__class__=OwnedPenFeatures;f.fixed_frame_cache=False
    return f


class FixedFrames(unittest.TestCase):
    def compare(self,f):
        f.fixed_frame_cache=False;p=f.policy();c=f.critic(p);r=f.reference_pose_world()
        f.fixed_frame_cache=True
        torch.testing.assert_close(f.policy(),p,atol=2e-7,rtol=0,equal_nan=True)
        torch.testing.assert_close(f.critic(f.policy()),c,atol=2e-7,rtol=0,equal_nan=True)
        torch.testing.assert_close(f.reference_pose_world(),r,atol=1e-7,rtol=0,equal_nan=True)

    def test_all_mock_phases_clamps_origins_and_live_suffix(self):
        f=feature_fixture()
        for phase in [-2,0,1,2,99]:
            f.motion.time_steps.fill_(phase);self.compare(f)
        f.env.sim.data.qpos[:,0]+=.025;f.env.sim.data.qpos[:,7:]+=.1;f.env.sim.data.qvel+=.17;f.action._raw_actions+=.3
        self.compare(f)

    def test_nonunit_mount_normalization_chains_remain_distinct_and_correct(self):
        f=fixture();f.env.sim.data.xquat[:,0]=torch.tensor([[.7,.2,.1,.3],[.11,.23,.37,.61]])
        f.reference=f.motion._frames;f.action._processed_actions=torch.zeros(2,20);f.env.sim.mj_model=SimpleNamespace(jnt_dofadr=np.array([0]),body_jntadr=np.array([-1,0]));f.reference_cache=FrozenPenReferenceCache(f);f.__class__=OwnedPenFeatures;f.fixed_frame_cache=False
        self.compare(f)

    def test_invalid_current_quaternion_is_not_repaired(self):
        f=feature_fixture();f.env.sim.data.qpos[0,3:7]=0;self.compare(f)
        self.assertTrue(torch.isnan(f.policy()[0,132:141]).any())

    def test_each_new_matrix_is_immutable_guarded(self):
        for name in ['policy_WT','critic_WT']:
            f=feature_fixture();getattr(f.reference_cache,name)[0,0,0]+=.01
            with self.assertRaises(ValueError):f.reference_cache.validate(f)

    def test_dynamic_feature_JVP_is_preserved(self):
        f=feature_fixture();inputs=(f.env.sim.data.qpos.clone(),f.env.sim.data.qvel.clone(),f.action._raw_actions.clone(),f.action._processed_actions.clone());tangents=tuple(torch.linspace(-.02,.03,x.numel()).reshape_as(x) for x in inputs)
        def compute(mode,*args):
            f.fixed_frame_cache=mode;f.env.sim.data.qpos,f.env.sim.data.qvel,f.action._raw_actions,f.action._processed_actions=args
            p=f.policy();return torch.cat([p,f.critic(p)],-1)
        left,lj=torch.autograd.functional.jvp(lambda *x:compute(False,*x),inputs,tangents)
        right,rj=torch.autograd.functional.jvp(lambda *x:compute(True,*x),inputs,tangents)
        torch.testing.assert_close(left,right,atol=2e-7,rtol=0);torch.testing.assert_close(lj,rj,atol=2e-6,rtol=0)

    def test_feature_switch_rejects_during_collection_or_nonboolean(self):
        runtime=OwnedPenRuntime.__new__(OwnedPenRuntime);runtime._collection_open=True
        with self.assertRaises(ValueError):runtime.set_fixed_frame_features(True)
        runtime._collection_open=False
        with self.assertRaises(ValueError):runtime.set_fixed_frame_features(1)

    def test_current_source_mode_fork_changes_only_declared_fields(self):
        path=(('attr','features'),('attr','fixed_frame_cache'));metadata=dict(checkpoint_schema=CENTERED_SCHEMA,fixed_frame_features='LIVE',arm='SUPPORT');nm=dict(fixed_frame_features='LIVE',unchanged=1)
        state=dict(schema=CENTERED_SCHEMA,metadata=metadata,native=dict(schema=native.SCHEMA,metadata=nm,python=dict(scalars={path:False}),immutable_scalars={},sources={'unchanged':'receipt'},native={'data':torch.arange(3)}),modules={'x':torch.arange(5)},rng={'x':42})
        before=copy.deepcopy(state);expected={**metadata,'fixed_frame_features':'CACHED'};expected_nm={**nm,'fixed_frame_features':'CACHED'}
        changed,patches=fork_fixed_frame_mode(state,expected,expected_nm)
        self.assertEqual(len(patches),3);self.assertFalse(state['native']['python']['scalars'][path]);self.assertTrue(changed['native']['python']['scalars'][path])
        torch.testing.assert_close(state['modules']['x'],before['modules']['x']);torch.testing.assert_close(changed['native']['native']['data'],before['native']['native']['data']);self.assertEqual(changed['native']['sources'],before['native']['sources']);self.assertEqual(changed['rng'],before['rng'])
        with self.assertRaises(ValueError):fork_fixed_frame_mode(state,{**expected,'arm':'RESIDUAL'},expected_nm)

    def test_saved_mode_mismatch_rejects_before_state_writes(self):
        path=(('attr','features'),('attr','fixed_frame_cache'));state=dict(metadata={'fixed_frame_features':'CACHED'},python={'scalars':{path:True}})
        native.validate_saved_feature_mode(state)
        for value in [False,1,None]:
            state['python']['scalars'][path]=value
            with self.assertRaisesRegex(ValueError,'mode disagrees'):native.validate_saved_feature_mode(state)
