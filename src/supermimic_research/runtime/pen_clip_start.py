"""Clip-bound native initialization with unchanged observation, action, stepping and reset logic."""
import inspect,textwrap
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from .pen_native_policy import NativePenPolicyAdapter
from ..artifacts import verify_file_pin


def install_start(runtime,d):
    c=runtime.cfg
    with np.load(verify_file_pin(c['clip_pin'])) as f:q=f['reference_qpos']
    with np.load(verify_file_pin(c['fixed_reference_pin'])) as f:pose=np.r_[f['COM'][0],f['quat_wxyz'][0]]
    with np.load(verify_file_pin(c['projected_pin'])) as f:projected=f['initial_hand_q']
    with np.load(verify_file_pin(c['control_pin'])) as f:ctrl=f['ctrl']
    ids=runtime.features.qids;qa=runtime.features.qa
    assert len(q)==c['frame_count'] and np.array_equal(runtime.motion._frames['joint_pos'].cpu().numpy(),q.astype(np.float32))
    expected=q[0].astype(np.float32);start=d.qpos[0].detach().cpu().numpy()
    np.testing.assert_array_equal(start[ids.cpu().numpy()],expected)
    np.testing.assert_allclose(start[qa:qa+3],pose[:3],rtol=0,atol=1e-6)
    if abs(start[qa+3:qa+7]@pose[3:])<1-1e-6:raise ValueError('Object start rotation mismatch')
    runtime.initial={k:getattr(d,k)[0].clone() for k in ('qpos','qvel','ctrl','qacc_warmstart','time','mocap_pos','mocap_quat')}
    def tensor(v):return torch.as_tensor(v,device=runtime.device,dtype=torch.float32)
    runtime.initial['qpos'][ids]=tensor(projected)
    for k in ('time','qvel','qacc_warmstart'):runtime.initial[k].zero_()
    runtime.initial_ema=tensor(ctrl-expected)
    runtime.initial_raw=2*runtime.initial_ema
    if runtime.initial_raw.abs().max()>1:raise ValueError('Start outside admissible action range')


@contextmanager
def clip_start_constructor():
    original=NativePenPolicyAdapter.__init__
    text=textwrap.dedent(inspect.getsource(original))
    guard="len(self.motion._frames['joint_pos'])!=369"
    if text.count(guard)!=1:raise ValueError('Upstream frame guard changed')
    text=text.replace(guard,"len(self.motion._frames['joint_pos'])!=cfg['frame_count']")
    prefix=text.index('    proposal=json.loads(')
    suffix=text.index('    self.pending=',prefix)
    text=text[:prefix]+'    install_start(self,self.env.sim.data)\n'+text[suffix:]
    space={}
    exec(compile(text,__file__,'exec'),dict(original.__globals__,install_start=install_start),space)
    with patch.object(NativePenPolicyAdapter,'__init__',space['__init__']):yield
