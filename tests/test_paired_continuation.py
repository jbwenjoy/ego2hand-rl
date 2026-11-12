import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.compound_response import CompoundResponseWM,fit_scales
from supermimic_research.learning.paired_response import paired_cache,paired_update
from supermimic_research.learning.paired_continuation import verify_next_update
from supermimic_research.learning.world_model import capture_rng
from supermimic_research.learning.checkpoint import _adam_valid


class PairedContinuationTests(unittest.TestCase):
    def test_actual_next_update_restores_both_batches_gradients_and_Adam(self):
        rng=np.random.default_rng(27);x=rng.normal(size=(8,102));y=rng.normal(size=(8,87));y[:,-3:]=abs(y[:,-3:]);scales=fit_scales(x,y)
        config=dict(width=16,learning_rate=.001,weight_decay=.0001,batch_size=4,pair_batch_size=2,gradient_clip_norm=10.,response_reference_rad=.025)
        model=CompoundResponseWM(scales,width=16);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=torch.Generator().manual_seed(8)
        paired=paired_cache(dict(x=x,y=y,pairs=np.array([[0,1,0,.001],[2,3,1,.01]])),model,'cpu');absolute={k:torch.tensor(v,dtype=torch.float32) for k,v in dict(x=x,y=y).items()}
        paired_update(model,opt,sampler,absolute,paired,config,coefficient=1.)
        payload=copy.deepcopy(dict(config=config,scales=scales,model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.get_state(),rng=capture_rng(device='cpu')))
        arm=dict(model=model,optimizer=opt,sampler=sampler,coefficient=1.)
        checks,losses=verify_next_update(arm,payload,absolute,paired,config,'cpu')
        self.assertTrue(all(checks.values()));self.assertEqual(len(checks),7);self.assertTrue(torch.isfinite(losses).all());self.assertTrue(_adam_valid(opt,model,2,torch))


if __name__=='__main__':unittest.main()
