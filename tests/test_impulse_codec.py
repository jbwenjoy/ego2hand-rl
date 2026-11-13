import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.impulse_codec import close_velocities,exp_world_unchecked,observed_defect_numpy
from supermimic_research.mechanics.response_decode import decode_response,momentum_defect


class ImpulseCodecTests(unittest.TestCase):
    def setUp(self):
        self.mass=torch.tensor(.25,dtype=torch.float64);Q=Rotation.from_rotvec([.4,-.3,.2]).as_matrix();self.I=torch.tensor(Q@np.diag([.001,.002,.003])@Q.T);self.inv=torch.linalg.inv(self.I);self.g=torch.tensor([0.,0.,-9.81],dtype=torch.float64)
        self.x=torch.zeros((3,102),dtype=torch.float64);R=Rotation.from_rotvec([.3,.2,.5]).as_matrix();self.x[:,43:49]=torch.tensor(np.r_[R[:,0],R[:,1]]);self.x[:,49:55]=torch.tensor([.1,-.2,.3,.4,-.5,.6])
        self.h=torch.zeros((3,87),dtype=torch.float64);self.h[:,43:46]=torch.tensor([[0.,0.,0.],[.02,-.03,.04],[-.1,.02,.03]]);self.h[:,78:84]=torch.tensor([.4,.5,2.,.001,-.002,.003])
    def test_independent_momentum_roundtrip_with_nonzero_native_defect(self):
        d=torch.tensor([[1e-7,-2e-7,3e-7,1e-9,-2e-9,3e-9]],dtype=torch.float64).repeat(3,1)
        p=close_velocities(self.x,self.h,self.mass,self.I,self.inv,self.g,d).numpy()
        recovered=observed_defect_numpy(self.x.numpy(),p,float(self.mass),self.I.numpy(),self.g.numpy())
        np.testing.assert_allclose(recovered,d.numpy(),atol=2e-17)
        p2=close_velocities(self.x,torch.tensor(p),self.mass,self.I,self.inv,self.g,torch.tensor(recovered)).numpy();np.testing.assert_allclose(p2,p,atol=3e-15)
    def test_wrench_to_velocity_derivative_and_rotation_zero_are_finite(self):
        h=self.h.clone().requires_grad_();v=torch.zeros_like(h);v[:,78]=1
        _,jvp=torch.func.jvp(lambda z:close_velocities(self.x,z,self.mass,self.I,self.inv,self.g),(h,),(v,))
        R=Rotation.from_rotvec([.3,.2,.5]).as_matrix();np.testing.assert_allclose(jvp[:,46:49].detach().numpy(),np.tile(.02/.25*R[:,0],(3,1)),atol=1e-14)
        self.assertTrue(torch.autograd.gradcheck(lambda r:exp_world_unchecked(r),torch.zeros((1,3),dtype=torch.float64,requires_grad=True)))
        self.assertTrue(torch.autograd.gradcheck(lambda z:close_velocities(self.x,z,self.mass,self.I,self.inv,self.g),h,atol=1e-6,rtol=1e-4))
    def test_world_left_exp_matches_independent_scipy(self):
        r=torch.tensor([[0.,0.,0.],[1e-9,-2e-9,3e-9],[.2,-.4,.5]],dtype=torch.float64)
        np.testing.assert_allclose(exp_world_unchecked(r).numpy(),Rotation.from_rotvec(r.numpy()).as_matrix(),atol=3e-16)


if __name__=='__main__':unittest.main()
