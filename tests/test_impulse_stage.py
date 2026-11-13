import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.compound_response import fit_scales
from supermimic_research.learning.impulse_response import ImpulseResponseWM
from supermimic_research.learning.impulse_stage import impulse_payload,verify_impulse_step
from supermimic_research.mechanics.impulse_codec import close_velocities


class ImpulseStageTests(unittest.TestCase):
    def test_new_head_optimizer_next_step_restore_and_label_only_defect(self):
        rng=np.random.default_rng(33);x=rng.normal(size=(8,102))*.01;x[:,43:49]=[1,0,0,0,1,0];x[:,-1]=.0003125;y=rng.normal(size=(8,87))*.001;y[:,-3:]=abs(y[:,-3:]);scales=fit_scales(x,y)
        head_scales=dict(head_mean=scales['y_mean'].copy(),head_scale=scales['y_scale'].copy());head_scales['head_mean'][46:52]=0;head_scales['head_scale'][46:52]=1e-7
        physics=dict(mass=.25,inertia_BODY=np.diag([.001,.002,.003]),gravity=np.array([0,0,-9.81]));config=dict(width=12,learning_rate=.001,weight_decay=.0001,batch_size=4,gradient_clip_norm=10.,auxiliary_weight=.1)
        model=ImpulseResponseWM(scales,head_scales,physics,width=12,coupled=True);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=torch.Generator().manual_seed(19)
        heads=y.copy();heads[:,46:52]=0
        cache=dict(x=torch.tensor(x,dtype=torch.float32),y=torch.tensor(y),IMPULSE=torch.tensor(heads));arm=dict(model=model,optimizer=opt,sampler=sampler,mode='IMPULSE')
        payload=copy.deepcopy(impulse_payload(arm,config,scales,head_scales,physics,update=0,device='cpu'));checks,values=verify_impulse_step(arm,payload,cache,config,'cpu');self.assertTrue(all(checks.values()));self.assertTrue(torch.isfinite(values).all())
        # Inference accepts only current102, not observed future residual labels.
        a=model.prediction(cache['x']);cache['IMPULSE']+=100;b=model.prediction(cache['x']);torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
