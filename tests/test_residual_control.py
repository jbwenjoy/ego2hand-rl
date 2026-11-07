from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.residual_control import ResidualEMAController,residual_ema
from supermimic_research.mechanics.compound_transition import observe55


class ResidualControlTests(unittest.TestCase):
    def test_near_bound_still_has_inward_authority_and_restore_is_exact(self):
        base=np.full(20,.999);a=ResidualEMAController(base,np.zeros(20),np.ones(20));b=ResidualEMAController(base,np.zeros(20),np.ones(20))
        first=a.step(np.ones(20));self.assertTrue(np.all(first['actual_control20']<=1));self.assertAlmostEqual(first['EMA_after20'][0],.0005)
        b.restore(a.state_dict());np.testing.assert_array_equal(a.step(-np.ones(20))['actual_control20'],b.step(-np.ones(20))['actual_control20']);self.assertGreater(a.negative[0],a.positive[0])

    def test_torch_shared_algebra_and_gradient_are_finite(self):
        import torch
        a=torch.tensor([[-.8,.4]],dtype=torch.float64,requires_grad=True);before=torch.tensor([[.01,-.01]],dtype=torch.float64);neg=torch.tensor([.05,.02]);pos=torch.tensor([.01,.05])
        result=residual_ema(a,before,neg,pos);result.sum().backward()
        np.testing.assert_allclose(result.detach().numpy(),residual_ema(a.detach().numpy(),before.numpy(),neg.numpy(),pos.numpy()));self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):ResidualEMAController(np.zeros(20),np.zeros(20),np.ones(20)).step(np.full(20,1.01))

    def test_POST_observer_COM_lever_arm_and_column_order(self):
        q=np.zeros(27);q[23]=1;q[20:23]=[1,2,3];v=np.zeros(26);v[23:26]=[0,0,2]
        result=observe55(q,v,np.array([.02,0,0]));np.testing.assert_allclose(result[40:43],[1.02,2,3]);np.testing.assert_allclose(result[43:49],[1,0,0,0,1,0]);np.testing.assert_allclose(result[49:52],[0,.04,0])


if __name__=='__main__':unittest.main()
