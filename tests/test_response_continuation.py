"""Optimizer continuation, native chart and action derivative checks."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.compound_response import CompoundResponseWM,fit_scales,INPUT_SCHEMA,OUTPUT_SCHEMA
from supermimic_research.learning.response_training import restore_response,response_update,response_payload
from supermimic_research.learning.response_diagnostics import paired_indices,vector_comparison,probe_metrics
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.mechanics.transition_records import transition_xy


class ResponseContinuationTests(unittest.TestCase):
    def test_optimizer_and_next_random_batch_replay(self):
        rng=np.random.default_rng(11);x=torch.tensor(rng.normal(size=(8,102)),dtype=torch.float32);y=torch.tensor(rng.normal(size=(8,87)),dtype=torch.float32);y[:,-3:]=y[:,-3:].abs()
        config=dict(width=12,learning_rate=.001,weight_decay=.0001,batch_size=4,gradient_clip_norm=10.)
        model=CompoundResponseWM(fit_scales(x.numpy(),y.numpy()),width=12);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=torch.Generator().manual_seed(3)
        response_update(model,opt,sampler,x,y,config)
        parent=dict(schema='compound_H20_response_checkpoint:1',planned_updates=1000,config=config,scales=fit_scales(x.numpy(),y.numpy()),normalizer_fit_role='TRAIN',dataset_pins={},packed={},model_input_schema=INPUT_SCHEMA,model_output_schema=OUTPUT_SCHEMA)
        state=copy.deepcopy(response_payload(model,opt,sampler,parent,update=1,device='cpu'))
        clone,clone_opt,clone_sampler=restore_response(state,'cpu')
        a,ia,ga=response_update(model,opt,sampler,x,y,config,capture_gradient=True);b,ib,gb=response_update(clone,clone_opt,clone_sampler,x,y,config,capture_gradient=True)
        for left,right in [(a,b),(ia,ib),(ga,gb),(model.state_dict(),clone.state_dict()),(opt.state_dict(),clone_opt.state_dict())]:self.assertTrue(raw_tree_equal(left,right))

    def test_world_left_rotation_chart_and_wrench_order(self):
        from scipy.spatial.transform import Rotation
        s=np.zeros(55);n=s.copy();R0=Rotation.from_rotvec([.5,.2,0]).as_matrix();s[43:49]=np.r_[R0[:,0],R0[:,1]];R=Rotation.from_rotvec([0,0,.02]).as_matrix()@R0;n[43:49]=np.r_[R[:,0],R[:,1]];n[40:43]=[.001,.002,.003]
        t=dict(source55=s,next55=n,warmstart26_before=np.zeros(26),warmstart26_after=np.ones(26),decision=dict(actual_control20=np.zeros(20)),mean_native_wrench_source_BODY_COM=np.arange(6),mean_native_group_Fn_N=np.arange(3))
        x,y=transition_xy(t,.0003125);self.assertEqual(x.shape,(102,));self.assertEqual(y.shape,(87,));np.testing.assert_allclose(y[43:46],[0,0,.02],atol=1e-14);np.testing.assert_array_equal(y[78:84],np.arange(6))

    def test_reject_response_pairs_with_different_source(self):
        raw=dict(k=np.array([0,0]),dt=np.array([.1,.1]),joint=np.array([0,0]),sign=np.array([-1,1]),x=np.zeros((2,102)))
        self.assertEqual(paired_indices(raw)[0].tolist(),[[0,1]]);raw['x'][1,0]=1
        with self.assertRaises(ValueError):paired_indices(raw)
        self.assertIsNone(vector_comparison(np.ones(3),np.zeros(3))['versus_zero_NMSE'])

    def test_native_secant_and_AD_chain_rule_on_known_linear_response(self):
        class LinearResponse(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('y_scale',torch.ones(87));self.register_buffer('matrix',torch.arange(20*87,dtype=torch.float32).reshape(20,87)/1024)
            def prediction(self,x):return x[:,81:101]@self.matrix
        model=LinearResponse();rows=[]
        for k in [0,3,6]:
            for dt in [.0003125,.00015625]:
                for joint in [0,4,8,9]:
                    for amplitude in [.01,.05,.2]:
                        for sign in [-1,1]:
                            x=np.zeros(102);x[81+joint]=sign*amplitude*.25;x[-1]=dt;y=x[81:101]@model.matrix.numpy()
                            rows.append(dict(x=x,y=y,k=k,dt=dt,joint=joint,amplitude=amplitude,sign=sign,du_da=.25))
        raw={key:np.array([r[key] for r in rows]) for key in rows[0]};metrics,_=probe_metrics(model,raw,'cpu')
        for item in metrics['by_amplitude'].values():self.assertLess(item['delta52']['AD']['versus_zero_NMSE'],1e-12)
        raw['du_da']*=2
        with self.assertRaises(ValueError):probe_metrics(model,raw,'cpu')


if __name__=='__main__':unittest.main()
