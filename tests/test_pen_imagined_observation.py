import unittest

import torch

from supermimic_research.mechanics.impulse_codec import exp_world_unchecked
from supermimic_research.mechanics.pen_imagined_observation import pen_imagined_observation


class TestPenImaginedObservation(unittest.TestCase):
    def test_broadcast_reference_and_nonidentity_wrist(self):
        dtype = torch.float64
        state = torch.zeros(2, 55, dtype=dtype)
        # Identity object orientation and hand pose in a rotated/translated wrist.
        state[:, 43] = state[:, 47] = 1
        state[:, :20] = .5
        state[:, 20:40] = 1.3
        state[:, 40:43] = torch.tensor([1.05, 2, 3], dtype=dtype)
        mount = torch.tensor([1, 2, 3, 2**-.5, 0, 0, 2**-.5], dtype=dtype)
        reference_pose = torch.tensor([1, 2.05, 3, 2**-.5, 0, 0, 2**-.5], dtype=dtype)
        ref = torch.arange(20, dtype=dtype)
        links = torch.arange(63, dtype=dtype).reshape(21,3)
        previous = torch.arange(40, dtype=dtype).reshape(2,20)/40
        obs = pen_imagined_observation(state,previous,ref,reference_pose,links,mount,
                                       .1+ref*0,.2+ref*0,.3+ref*0)
        pos = .15*torch.tanh(torch.tensor(.05/.15,dtype=dtype)).item()
        ref9 = torch.tensor([pos,0,0,1,0,0,0,1,0],dtype=dtype)
        current9 = torch.tensor([0,-pos,0,0,-1,0,1,0,0],dtype=dtype)
        for i,row in enumerate(obs):
            expected=torch.cat((ref,ref9,links.flatten(),ref*0+.4,ref*0+.25,current9,previous[i]))
            torch.testing.assert_close(row,expected)

    def test_pose_and_action_feedback_gradients(self):
        dtype = torch.float64
        ref = torch.zeros(20,dtype=dtype)
        mount = torch.tensor([.1,0,.3,1,0,0,0],dtype=dtype)
        vec = torch.tensor([.1,-.3,.7],dtype=dtype,requires_grad=True)
        pos = torch.tensor([.21,-.11,.15],dtype=dtype,requires_grad=True)
        raw = torch.linspace(-.8,.8,20,dtype=dtype).requires_grad_(True)

        def observe(vec,pos,raw):
            R = exp_world_unchecked(vec)
            state = torch.cat([ref,ref,pos,R[:,0],R[:,1],torch.zeros(6,dtype=dtype)])
            return pen_imagined_observation(state,raw,ref,mount,torch.zeros(63,dtype=dtype),
                                            mount,ref,ref,ref)

        self.assertTrue(torch.autograd.gradcheck(observe,(vec,pos,raw),fast_mode=True))
        d_rot,d_pos,d_raw=torch.autograd.grad(observe(vec,pos,raw)[132:].sum(),(vec,pos,raw))
        self.assertGreater(float(d_rot.norm()),.01)
        self.assertGreater(float(d_pos.norm()),.01)
        torch.testing.assert_close(d_raw,torch.ones_like(raw))


if __name__ == "__main__":
    unittest.main()
