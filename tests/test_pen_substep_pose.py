import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.pen_substep_pose import cv_qpos,SubstepPose


class PenSubstepPoseTest(unittest.TestCase):
    def test_world_rotation_and_zero_source_residual(self):
        state=torch.zeros(55,dtype=torch.double)
        initial=Rotation.from_rotvec([.2,-.3,1.2])
        state[43:46]=torch.from_numpy(initial.as_matrix()[:,0])
        state[46:49]=torch.from_numpy(initial.as_matrix()[:,1])
        state[49:55]=torch.tensor([.7,1.,-.6,-.4,.8,1.1],dtype=state.dtype)
        correction=torch.zeros(7,26,dtype=state.dtype)
        correction[:,23:]=torch.tensor([.009,-.015,.023],dtype=state.dtype)
        pred=cv_qpos(state,correction).numpy()
        t=np.arange(8)*.0025
        np.testing.assert_allclose(pred[:,:3],t[:,None]*state[49:52].numpy())
        expect=Rotation.from_rotvec(t[:,None]*state[52:55].numpy())*initial
        extra=Rotation.from_rotvec(np.row_stack([np.zeros(3),correction[:,23:].numpy()]))
        actual=Rotation.from_quat(pred[:,[4,5,6,3]])
        np.testing.assert_allclose((actual*(extra*expect).inv()).magnitude(),0,atol=1e-12)

    def test_pd_gradient_batched_matches_finite_difference(self):
        torch.manual_seed(10)
        net=SubstepPose().double().eval().requires_grad_(False)
        s=torch.randn(2,55,dtype=torch.double)
        r=Rotation.from_rotvec([[.1,.2,.3],[.2,-.1,-.4]]).as_matrix()
        s[:,43:46]=torch.from_numpy(r[:,:,0]);s[:,46:49]=torch.from_numpy(r[:,:,1])
        w=torch.randn(2,26,dtype=torch.double);u=torch.randn(2,20,dtype=torch.double)
        v=torch.randn_like(u)
        calc=lambda p:net.predict_qpos(s,w,p)
        _,jvp=torch.autograd.functional.jvp(calc,u,v)
        numerical=(calc(u+1e-5*v)-calc(u-1e-5*v))/(2e-5)
        torch.testing.assert_close(jvp,numerical,atol=1e-8,rtol=1e-6)
        torch.testing.assert_close(jvp[:,0],torch.zeros_like(jvp[:,0]))
        self.assertGreater(float(jvp[:,1:].norm()),.001)


if __name__=='__main__':unittest.main()
