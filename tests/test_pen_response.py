import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_response import fit_pen_scales,PenResponseWM


class PenResponseTests(unittest.TestCase):
    def test_fn6_head_and_impulse_closure_preserve_full_pen_schema(self):
        x=np.zeros((2,102));x[:,43]=1;x[:,47]=1;x[:,-1]=.005
        y=np.zeros((2,90));y[:,80]=.03*9.81;y[:,84]=.3
        scales,hs=fit_pen_scales(x,y,y.copy());physics=dict(mass=.03,inertia_BODY=np.diag([1e-6,2e-4,2e-4]),gravity=[0,0,-9.81])
        model=PenResponseWM(scales,hs,physics,width=16)
        out=model.prediction(torch.tensor(x,dtype=torch.float32))
        self.assertEqual(out.shape,(2,90));self.assertTrue(torch.isfinite(out).all());self.assertTrue((out[:,84:]>=0).all())
        np.testing.assert_allclose(out.detach().numpy()[:,49:52],0,atol=1e-6)
