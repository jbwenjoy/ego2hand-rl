import copy
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import numpy as np
import torch
from supermimic_research.learning.pen_response import fit_pen_scales,PenResponseWM
from supermimic_research.learning.pen_path_response import ARMS,fit_path_units,update_path
from supermimic_research.learning.pen_path_fit import new_arm,save_arm,restore_arm,health,fixed_spec
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.learning.pen_path_reward import scores,gradients


class PathResponseTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(11);x=np.zeros((6,102));x[:,43]=x[:,47]=1;x[:,-1]=.0025;x[:,81]=[.002,.002,-.002,-.002,.004,-.004]
        y=rng.normal(0,.001,(6,90));y[:,84:]=abs(y[:,84:]);scales,_=fit_pen_scales(x[:4],y[:4],y[:4].copy())
        labels=np.array([.10,.11,.12,.13,100.,200.]);mask=np.array([1,1,1,1,0,0],bool)
        groups=[dict(rows={'1':[0,1],'-1':[2,3]},epsilon=e,complete=True,qualified=True) for e in [.002,.005]]
        groups.append(dict(rows={'1':[4,5],'-1':[0,1]},epsilon=.002,complete=True,qualified=False))
        units=fit_path_units(labels,mask,groups,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        cfg=dict(width=16,batch_size=4,pair_batch_size=2,learning_rate=.0003,weight_decay=.00001,gradient_clip_norm=1.,auxiliary_weight=.1,sampler_seed=14,pair_sampler_seed=15)
        context=dict(config=cfg,backend=dict(backend_id='FIXTURE_ONLY'),physics=dict(mass=.03,inertia_BODY=np.diag([1e-6,2e-4,2e-4]),gravity=[0,0,-9.81]),scales=scales,response_scale=np.ones(90),path_units=units,contract=dict(fixture=True))
        t=lambda a:torch.tensor(a,dtype=torch.float64)
        cache=dict(x=torch.tensor(x[:4],dtype=torch.float32),y=t(y[:4]),path=t(labels[:4]),x_plus=torch.tensor(x[[0,1]],dtype=torch.float32),x_minus=torch.tensor(x[[2,3]],dtype=torch.float32),
            epsilon=t([.002,.005]),finite_contrast=t((y[[0,1]]-y[[2,3]])/np.array([.004,.01])[:,None]),response_scale=torch.ones(90,dtype=torch.float64),path_contrast=t([-.02/.004,-.02/.01]),contract=context['contract'])
        return context,cache,labels,mask,groups

    def test_direct_prediction_and_point_loss_reproduce_maintained_math(self):
        context,cache,*_=self.fixture();a=new_arm('RESPONSE_ONLY',context,'cpu')['model']
        old=PenResponseWM(context['scales'],dict(head_mean=context['scales']['y_mean'],head_scale=context['scales']['y_scale']),context['physics'],width=16,coupled=False);old.net.load_state_dict(a.net.state_dict())
        self.assertTrue(torch.equal(a.prediction(cache['x']),old.prediction(cache['x'])))
        point,_=a.point_terms(cache['x'],cache['y'],cache['path'],.1)
        reference,_=old.supervised_loss(cache['x'],cache['y'],cache['y'],auxiliary_weight=.1)
        self.assertTrue(torch.equal(point,reference))

    def test_train_units_ignore_invalid_labels_and_reject_validation(self):
        _,_,labels,mask,groups=self.fixture();a=fit_path_units(labels,mask,groups,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        labels[~mask]+=1e10;b=fit_path_units(labels,mask,groups,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        self.assertTrue(all(np.array_equal(v,b[k]) for k,v in a.items()))
        with self.assertRaises(ValueError):fit_path_units(labels,mask,groups,role='VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT')
        groups[0]['qualified']=False
        with self.assertRaises(ValueError):fit_path_units(labels,mask,groups,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')

    def test_all_arms_restore_next_update_and_independent_adam_storage(self):
        context,cache,*_=self.fixture()
        for name in ARMS:
            arm=new_arm(name,context,'cpu');update_path(arm,cache,context['config']);saved=copy.deepcopy(save_arm(arm,context,1,'cpu'));before=copy.deepcopy(saved)
            left,right=restore_arm(saved,context,'cpu'),restore_arm(saved,context,'cpu')
            self.assertNotEqual(next(iter(left['optimizer'].state.values()))['exp_avg'].data_ptr(),next(iter(right['optimizer'].state.values()))['exp_avg'].data_ptr())
            a=update_path(left,cache,context['config'],capture_gradient=True);b=update_path(right,cache,context['config'],capture_gradient=True)
            self.assertTrue(raw_tree_equal(a,b));self.assertTrue(raw_tree_equal(left['model'].state_dict(),right['model'].state_dict()));self.assertTrue(raw_tree_equal(left['optimizer'].state_dict(),right['optimizer'].state_dict()));self.assertTrue(raw_tree_equal(before,saved))

    def test_foreign_context_and_cursor_reject_before_construction(self):
        context,_,*_=self.fixture();saved=save_arm(new_arm('PATH_POINT',context,'cpu'),context,0,'cpu')
        for defect in ['schema','context','cursor','path_schema']:
            bad=copy.deepcopy(saved)
            if defect=='context':bad['context']['path_units']['point_scale']*=2
            elif defect=='cursor':bad['update']=True
            else:bad[defect]='foreign'
            with self.subTest(defect=defect),patch('supermimic_research.learning.pen_path_fit.new_arm',side_effect=AssertionError('constructed')):
                with self.assertRaises(ValueError):restore_arm(bad,context,'cpu')

    def test_normalization_buffer_change_rejects(self):
        context,_,*_=self.fixture();saved=copy.deepcopy(save_arm(new_arm('PATH_POINT',context,'cpu'),context,0,'cpu'));saved['model']['path_point_scale']*=2
        with self.assertRaises(ValueError):restore_arm(saved,context,'cpu')

    def test_baseline_path_head_and_moments_stay_zero_with_owned_cursor(self):
        context,cache,*_=self.fixture();arm=new_arm('RESPONSE_ONLY',context,'cpu')
        for _ in range(2):update_path(arm,cache,context['config'])
        for p in arm['model'].path_head.parameters():
            self.assertFalse(bool(p.any()));state=arm['optimizer'].state[p];self.assertEqual(float(state['step']),2);self.assertFalse(bool(state['exp_avg'].any()));self.assertFalse(bool(state['exp_avg_sq'].any()))

    def test_health_preserves_live_model_optimizer_and_streams(self):
        context,cache,*_=self.fixture();arms={n:new_arm(n,context,'cpu') for n in ARMS}
        for a in arms.values():update_path(a,cache,context['config'])
        before=copy.deepcopy({n:save_arm(a,context,1,'cpu') for n,a in arms.items()});work=dict(QA_attempted_calls=0,QA_completed_steps=0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ck.pt';torch.save(dict(update=1,planned_updates=2000,arms=before),p);checks=health(p,arms,context,cache,1,'cpu',work)
        self.assertTrue(all(checks.values()),[k for k,v in checks.items() if not v]);self.assertEqual(work,dict(QA_attempted_calls=6,QA_completed_steps=6))
        for n,a in arms.items():
            self.assertTrue(raw_tree_equal(before[n]['model'],a['model'].state_dict()));self.assertTrue(raw_tree_equal(before[n]['optimizer'],a['optimizer'].state_dict()));self.assertTrue(raw_tree_equal(before[n]['sampler'],a['sampler'].get_state()));self.assertTrue(raw_tree_equal(before[n]['pair_sampler'],a['pair_sampler'].get_state()))

    def test_registered_budget_and_validation_schedule_fixed(self):
        cfg=tomllib.loads((Path(__file__).resolve().parents[1]/'configs/pen_path_fit_v1.toml').read_text());self.assertEqual(fixed_spec(cfg)['models'],ARMS)
        for delta in [dict(planned_updates=100),dict(models=ARMS[:1]),dict(validation_evaluation_updates=[100,2000]),dict(auxiliary_weight=True)]:
            with self.assertRaises(ValueError):fixed_spec({**cfg,**delta})

    def test_added_path_chain_includes_controller_gain_and_direct_raw_cost(self):
        class Toy:
            def full_prediction(self,x):return x[:,:90],1+3*x[:,81:101].mean(-1)
        def proxy(x,p,raw,*_):
            v=2*p[:,81:90].sum(-1)-.02*raw.square().sum(-1)
            return dict(observable_residual=v,observable_support=v+raw[:,0]*.03)
        context=dict(EMA=torch.full((1,20),.1,dtype=torch.float64),reference_joints=torch.zeros(1,20,dtype=torch.float64),encoder_bias=torch.zeros(1,20,dtype=torch.float64),lower=torch.full((1,20),-2.),upper=torch.full((1,20),2.))
        x=torch.zeros(1,102,dtype=torch.float64);x[:,81:101]=.1;raw=torch.full((1,20),.2,dtype=torch.float64);direction=torch.linspace(-.2,.3,20,dtype=torch.float64)
        with patch('supermimic_research.learning.pen_path_reward.observable_proxy',side_effect=proxy):
            g=gradients(Toy(),x,raw,context,{},dict(geometry_soft_weight=.25))
            epsilon=1e-5;xp,xm=x.clone(),x.clone();xp[:,81:101]+=.25*epsilon*direction;xm[:,81:101]-=.25*epsilon*direction
            vp=scores(Toy(),xp,raw+epsilon*direction,context,{},dict(geometry_soft_weight=.25))[0]
            vm=scores(Toy(),xm,raw-epsilon*direction,context,{},dict(geometry_soft_weight=.25))[0]
        torch.testing.assert_close(g['RAW']@direction,(vp-vm)/(2*epsilon),atol=1e-9,rtol=1e-8)
        self.assertTrue(bool((g['controller_diagonal']==.25).all()))


if __name__=='__main__':unittest.main()
