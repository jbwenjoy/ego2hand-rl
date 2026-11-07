from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.compound_response import fit_scales,CompoundResponseWM


class CompoundResponseTests(unittest.TestCase):
    def test_explicit_train_scales_and_nonnegative_native_force_head(self):
        import torch
        x=np.arange(8*102,dtype=float).reshape(8,102)*.001;y=np.zeros((8,87));scales=fit_scales(x,y)
        np.testing.assert_allclose(scales['x_mean'],x.mean(0));self.assertTrue(np.all(scales['x_scale']>0));self.assertTrue(np.all(scales['y_mean'][-3:]==0))
        model=CompoundResponseWM(scales,width=16);before={k:v.clone() for k,v in model.state_dict().items()};a=torch.tensor(x,dtype=torch.float32,requires_grad=True)
        prediction=model.prediction(a);self.assertTrue(torch.all(prediction[:,-3:]>=0));model.loss(a,torch.tensor(y,dtype=torch.float32)).backward();self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_no_batch_dependent_statistics_or_geometry_calls(self):
        import torch
        rng=np.random.default_rng(7);x=rng.normal(size=(7,102));y=rng.normal(size=(7,87));y[:,-3:]=abs(y[:,-3:])
        model=CompoundResponseWM(fit_scales(x,y),width=16);torch.nn.init.normal_(model.net[-1].weight,std=.01);a=torch.tensor(x,dtype=torch.float32)
        torch.testing.assert_close(model(a),torch.cat([model(a[:3]),model(a[3:])]),atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
