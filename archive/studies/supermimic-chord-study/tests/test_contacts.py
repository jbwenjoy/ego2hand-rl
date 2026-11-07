import unittest
import numpy as np
import torch
from chord_study.contacts import sample_basis, wrench_support, support_reward
from chord_study.geometry import world_to_object


class ContactGeometry(unittest.TestCase):
    def setUp(self):
        self.p=torch.tensor([[.1,.02,0.],[-.1,0.,0.]],dtype=torch.float64)
        self.n=torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
        self.active=torch.ones(2,dtype=torch.bool)
        self.basis=sample_basis(64,seed=17,dtype=torch.float64)

    def support(self,p=None,n=None,mask=None,r=.12):
        return wrench_support(self.p if p is None else p,self.n if n is None else n,
                              self.active if mask is None else mask,self.basis,r)

    def test_no_contact_and_empty_contact_are_zero(self):
        self.assertEqual(self.support(mask=torch.zeros(2)).abs().sum().item(),0)
        empty=wrench_support(self.p[:0],self.n[:0],self.active[:0],self.basis,.12)
        self.assertEqual(empty.abs().sum().item(),0)

    def test_permutation_and_duplicate_invariance(self):
        torch.testing.assert_close(self.support(),self.support(self.p.flip(0),self.n.flip(0)))
        dup=wrench_support(self.p.repeat(2,1),self.n.repeat(2,1),self.active.repeat(2),self.basis,.12)
        torch.testing.assert_close(dup,self.support())

    def test_uniform_scaling_divides_torque_once(self):
        torch.testing.assert_close(self.support(),self.support(p=100*self.p,r=12))

    def test_point_gradient_matches_finite_difference(self):
        p=self.p.clone().requires_grad_(True)
        weights=torch.linspace(.2,1.,64,dtype=torch.float64)
        loss=(self.support(p=p)*weights).sum();loss.backward()
        eps=1e-6;plus=self.p.clone();minus=self.p.clone()
        plus[0,1]+=eps;minus[0,1]-=eps
        fd=((self.support(p=plus)-self.support(p=minus))*weights).sum()/(2*eps)
        torch.testing.assert_close(p.grad[0,1],fd,atol=1e-5,rtol=1e-5)

    def test_rigid_world_transform_recovered(self):
        # +90deg about z, followed by translation; object local points/normals unchanged.
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        trans=np.array([3.,2.,1.])
        pw=self.p.numpy()@rot.T+trans;nw=self.n.numpy()@rot.T
        pose=np.r_[trans,np.sqrt(.5),0.,0.,np.sqrt(.5)]
        po,no=world_to_object(pw,nw,pose)
        np.testing.assert_allclose(po,self.p.numpy(),atol=1e-12)
        np.testing.assert_allclose(no,self.n.numpy(),atol=1e-12)

    def test_bad_normal_and_soft_observed_mask_fail(self):
        with self.assertRaisesRegex(ValueError,"normal"):
            self.support(n=torch.zeros_like(self.n))
        with self.assertRaisesRegex(ValueError,"binary"):
            self.support(mask=torch.full((2,),.5))

    def test_reward_empty_reference_zero(self):
        v=torch.zeros((2,1,1,64))
        torch.testing.assert_close(support_reward(v,v),torch.zeros(2))

    def test_reward_ignores_inactive_bodies_not_active_directions(self):
        ref=torch.zeros((1,2,2,3));ref[0,0,0,:]=1
        torch.testing.assert_close(support_reward(ref,ref),torch.ones(1))
        sim=ref.clone();sim[0,0,0,0]=0
        torch.testing.assert_close(support_reward(sim,ref),torch.tensor([2/3]))


if __name__=="__main__": unittest.main()
