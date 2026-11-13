import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.impulse_response import ImpulseResponseWM
from supermimic_research.learning.compound_response import fit_scales
from supermimic_research.learning.impulse_stage import update_impulse,impulse_payload,verify_impulse_step
from supermimic_research.learning.response_strata import response_strata
from supermimic_research.learning.checkpoint import _adam_valid


class ImpulseContinuationTests(unittest.TestCase):
    def test_nonzero_Adam_state_next_update_exact(self):
        rng=np.random.default_rng(48);x=rng.normal(size=(8,102))*.01;x[:,43:49]=[1,0,0,0,1,0];x[:,-1]=.0003125;y=rng.normal(size=(8,87))*.001;y[:,-3:]=abs(y[:,-3:]);scales=fit_scales(x,y)
        hs=dict(head_mean=scales['y_mean'].copy(),head_scale=scales['y_scale'].copy());hs['head_mean'][46:52]=0;hs['head_scale'][46:52]=1e-7
        physics=dict(mass=.25,inertia_BODY=np.diag([.001,.002,.003]),gravity=np.array([0,0,-9.81]));cfg=dict(width=12,learning_rate=.001,weight_decay=.0001,batch_size=4,gradient_clip_norm=10.,auxiliary_weight=.1)
        m=ImpulseResponseWM(scales,hs,physics,width=12,coupled=True);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);sampler=torch.Generator().manual_seed(14)
        target=y.copy();target[:,46:52]=0;cache=dict(x=torch.tensor(x,dtype=torch.float32),y=torch.tensor(y),IMPULSE=torch.tensor(target));arm=dict(model=m,optimizer=opt,sampler=sampler,mode='IMPULSE')
        update_impulse(arm,cache,cfg);saved=copy.deepcopy(impulse_payload(arm,cfg,scales,hs,physics,update=1,device='cpu'))
        checks,values=verify_impulse_step(arm,saved,cache,cfg,'cpu');self.assertTrue(all(checks.values()));self.assertTrue(torch.isfinite(values).all());self.assertTrue(_adam_valid(opt,m,2,torch))

    def test_component_report_keeps_zero_native_effect_and_empty_axes_explicit(self):
        raw=dict(pairs=np.array([[0,1,0,.01],[2,3,17,.006]]),amplitude=np.full(4,.04));native=np.zeros((2,87));native[0,46:49]=2;pred=np.ones((2,87));arrays=dict(native_secant_normalized=native,model_AD_normalized=pred,model_secant_normalized=pred)
        result=response_strata(raw,arrays);self.assertEqual(result['pairs'],2)
        zero=result['groups']['amplitude0.04/joint17']['components']['COM_velocity3']['AD']
        self.assertEqual(zero['native_RMS'],0);self.assertEqual(zero['predicted_RMS'],1);self.assertIsNone(zero['versus_zero_NMSE'])
        empty=result['groups']['amplitude0.04/joint19'];self.assertEqual(empty['pairs'],0);self.assertIsNone(empty['components'])


if __name__=='__main__':unittest.main()
