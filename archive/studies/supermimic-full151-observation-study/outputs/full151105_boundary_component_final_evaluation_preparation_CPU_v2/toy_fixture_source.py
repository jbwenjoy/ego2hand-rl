"""Exactly3 toy-Torch wrapper/loss contracts; no native WM/data/optimizer."""
import copy,hashlib,json,random,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_impulse_component105_v1 as core


def pin(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


class ToyNorm(nn.Module):
    def __init__(self,dim,std=None):
        super().__init__();self.eps=.001;self.max_count=100.;self._host_count=0.
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim) if std is None else std.clone());self.register_buffer('count',torch.tensor(0,dtype=torch.int64))
    def std(self):return self.scale
    def forward(self,x):return (x-self.mean)/self.scale


class ToyWM(nn.Module):
    """Own synthetic67->(52 delta,15) function; never native WorldModel."""
    def __init__(self):
        super().__init__();self.amp=False;self.calls=0
        scale=torch.ones(52);scale[46:49]=torch.tensor([2.,3.,4.]);scale[49:52]=torch.tensor([5.,6.,7.])
        self.norm=ToyNorm(298);self.delta_norm=ToyNorm(52,scale);self.register_buffer('error_floor',torch.full((52,),.05))
        self.net=nn.Sequential(nn.Linear(298,512),nn.Tanh(),nn.Linear(512,67))
        with torch.no_grad():self.net[-1].weight.mul_(.01);self.net[-1].bias.mul_(.01)
    def forward(self,x,u,dynamics):
        self.calls+=1;y=self.net(self.norm(torch.cat((dynamics,u),dim=-1)));d=y[:,:52]
        # Synthetic diagonal R change makes prediction-R gradient observable;
        # it is intentionally not a native LogR/integrate implementation.
        R=x[:,43:52].reshape(-1,3,3)+.01*torch.diag_embed(d[:,43:46])
        pred=torch.cat((x[:,:40]+.01*d[:,:40],x[:,40:43]+.01*d[:,40:43],R.flatten(1),x[:,52:55]+.01*d[:,46:49],x[:,55:58]+.01*d[:,49:52]),dim=-1)
        return pred,y[:,52:]


def make_inputs(n,xi):
    x=torch.zeros((n,58));x[:,43:52]=torch.eye(3).reshape(1,9);x[:,55:58]=torch.tensor([1.,2.,3.]);u=torch.full((n,20),.125)
    dynamics=torch.cat((x,xi[None].expand(n,-1),torch.zeros((n,174))),dim=-1)
    return x,u,dynamics


def main():
    torch.set_num_threads(4);out=ROOT/'outputs/full151105_boundary_impulse_component_preparation_CPU_v1';core.require(not out.exists(),'Fresh synthetic preparation');out.mkdir()
    pins={k:pin(v) for k,v in core.OWN.items()};pins['recipe']=pin(core.RECIPE);core.require(pins['recipe']['sha256']==core.RECIPE_SHA,'Unchanged5926 draft')
    write(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+('_source.md' if name=='recipe' else '_source.py'))).write_bytes(Path(value['path']).read_bytes())
    checks={};components={}
    def eq(v):core.require(bool(v),'Toy contract')
    def reject(fn):
        try:fn()
        except (ValueError,TypeError):return
        raise AssertionError('Expected safe API rejection')
    try:
        raw=ToyWM().eval();n=3;xi=torch.arange(46,dtype=torch.float32)/100
        std=torch.maximum(raw.delta_norm.std(),raw.error_floor)
        mass=torch.tensor(2.,dtype=torch.float32);Ib=torch.diag(torch.tensor([2.,3.,4.],dtype=torch.float64))
        g=torch.tensor([[0.,0.,-10.]],dtype=torch.float64).expand(n,-1).clone();dt=torch.full((n,),.02,dtype=torch.float64);P=torch.eye(3)[None].expand(n,-1,-1).clone()
        ctx=core.fixed_context_from_rows(mass,Ib,g,dt,P,std,xi)
        reject(lambda:core.fixed_context_from_rows(mass,Ib,g,dt+torch.tensor([0.,.01,0.],dtype=torch.float64),P,std,xi))
        x,u,dynamics=make_inputs(n,xi)
        baseline=raw(x,u,dynamics)
        for parameter in raw.parameters():parameter.requires_grad_(False)
        rng=torch.get_rng_state().clone();python_rng=random.getstate();raw_before=core.state_digest(raw)
        components,receipt=core.make_matched_components(raw,ctx)
        eq(torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate() and receipt['raw_WM_unchanged'] and receipt['disjoint_model_head_buffer_storage'])
        outputs={name:module(x,u,dynamics) for name,module in components.items()}
        for name,module in components.items():
            pred,self15,z,mode=outputs[name]
            eq(core.bit_equal(pred,baseline[0]) and core.bit_equal(self15,baseline[1]) and bool((z==0).all()) and bool((mode==0).all()))
            eq(module.counts['hook_calls']==module.counts['forward_returned']==1 and module._cache is None and not module._active)
            reject(lambda module=module:copy.deepcopy(module))
        eq(core.state_digest(raw)==raw_before and all(not p.requires_grad for p in raw.parameters()) and raw.calls==1)
        checks['native67_toy_function_bits_independent_clones_caches_scalar_RNG']=True

        method=components[core.ARMS[1]];z=torch.tensor([[[1.,2.,3.,1.,2.,3.],[-1.,.5,0.,0.,0.,-1.]]]).expand(n,-1,-1).clone();mode=torch.zeros((n,2))
        savedJ=torch.tensor([[[8.,12.,24.],[-4.,3.,0.]]],dtype=torch.float64).expand(n,-1,-1).clone()
        savedK=torch.tensor([[[10.,36.,84.],[0.,0.,-28.]]],dtype=torch.float64).expand(n,-1,-1).clone()
        labels={'J_palm_F64':savedJ,'K_palm_F64':savedK,'boundary_constraint_witness':torch.zeros((n,2),dtype=torch.bool),
            'r_v_saved_m_s_F64':torch.tensor([[.25,-.5,1.]],dtype=torch.float64).expand(n,-1).clone(),'r_w_saved_rad_s_F64':torch.tensor([[-.2,.3,.4]],dtype=torch.float64).expand(n,-1).clone()}
        point=torch.tensor(.125,dtype=torch.float32)
        loss=core.component_losses(method,x,x,z,mode,point,labels,lambda_mode=.2,lambda_imp=.3,lambda_cons=1.)
        expectedJ=torch.tensor([[[4.,12.,24.],[-4.,3.,0.]]],dtype=torch.float64).expand(n,-1,-1)
        expectedK=torch.tensor([[[10.,36.,84.],[0.,0.,-28.]]],dtype=torch.float64).expand(n,-1,-1)
        eq(torch.equal(loss['J_hat_palm_F64'],expectedJ) and torch.equal(loss['K_hat_palm_F64'],expectedK))
        eq(loss['L_state_FP32'] is point and loss['L_state_FP32'].dtype==torch.float32 and loss['L_imp_F64'].dtype==torch.float64)
        eq(torch.allclose(loss['L_imp_F64'],torch.tensor(1/12,dtype=torch.float64),rtol=0,atol=1e-15))
        altered={**labels,'boundary_constraint_witness':torch.ones((n,2),dtype=torch.bool)}
        other=core.component_losses(method,x,x,z,mode+10,point,altered,lambda_mode=.2,lambda_imp=.3,lambda_cons=1.)
        eq(torch.equal(loss['J_hat_palm_F64'],other['J_hat_palm_F64']) and torch.equal(loss['K_hat_palm_F64'],other['K_hat_palm_F64']) and torch.equal(loss['L_imp_F64'],other['L_imp_F64']))
        expectedrv=torch.tensor([[0.,-7.5,-11.8]],dtype=torch.float64).expand(n,-1);expectedrw=torch.tensor([[-5.,-12.,-14.]],dtype=torch.float64).expand(n,-1)
        eq(torch.allclose(loss['r_v_hat_F64'],expectedrv,rtol=0,atol=1e-14) and torch.allclose(loss['r_w_hat_F64'],expectedrw,rtol=0,atol=1e-14))
        expectedcons=torch.cat(((expectedrv-labels['r_v_saved_m_s_F64'])/std[46:49].double(),(expectedrw-labels['r_w_saved_rad_s_F64'])/std[49:52].double()),dim=-1).square().mean()
        eq(torch.allclose(loss['L_cons_F64'],expectedcons,rtol=0,atol=1e-14) and bool(loss['L_cons_F64']>0))
        checks['decoded_SI_effective52_zspace_no_mode_gating_saved_defect']=True

        before=core.state_digest(method);pred,self15,z,mode=method(x,u,dynamics)
        labels3={k:v.clone() for k,v in labels.items()};labels3['J_palm_F64'].zero_();labels3['K_palm_F64'].zero_()
        losses=core.component_losses(method,x,pred,z,mode,pred[:,:40].square().mean(),labels3,lambda_mode=.2,lambda_imp=.3,lambda_cons=1.)
        grad_pred,grad_head,grad_trunk=torch.autograd.grad(losses['L_cons_F64'],(pred,method.aux_head.weight,method.wm.net[0].weight),retain_graph=True)
        eq(bool(torch.isfinite(grad_pred).all()) and bool(grad_pred[:,43:52].abs().sum()>0) and bool(torch.isfinite(grad_head).all()) and bool(grad_head.abs().sum()>0) and bool(torch.isfinite(grad_trunk).all()) and bool(grad_trunk.abs().sum()>0))
        losses['L_total'].backward()
        eq(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in method.parameters()))
        eq(core.state_digest(method)==before and core.state_digest(raw)==raw_before);method.assert_frozen()
        eq(method._cache is None and not method._active and not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco' not in sys.modules)
        checks['F64_solve_nonzero_head_trunk_predR_gradients_frozen_buffers_zeroopt']=True
        for name,module in components.items():module.close();eq(not module.wm.net[-1]._forward_pre_hooks)
        for value in pins.values():eq(pin(value['path'])['sha256']==value['sha256'])
        report={'schema':'full151105_boundary_impulse_component_CPU_contract:1','status':'PASS_THREE_TOY_WRAPPER_AND_DIFFERENTIABLE_LOSS_CONTRACTS',
            'checks':checks,'passed':len(checks),'total':3,'source_pins':pins,'construction_receipt':receipt,
            'synthetic_scope':{'ToyWM_constructors':1,'Toy_native_forward_calls':4,'synthetic_autograd_grad_calls':1,'synthetic_backward_calls':1,'optimizer_steps':0},
            'actual_checkpoints_datasets_reads_nativeWorldModel_ctors_FW_optimizer_training_physics_GPU':0}
        write(out/'report.json',report)
        ready={'schema':'full151105_boundary_impulse_component_CPU_READY:1','status':report['status'],'source_pins':{k:v for k,v in pins.items() if k!='recipe'},'recipe_draft':pins['recipe'],
            'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},'api':{'factory':'make_matched_components(raw_wm,fixed_context)','context':'fixed_context_from_rows(...)','loss':'component_losses(...,lambda_mode=...,lambda_imp=...,lambda_cons=...)'},'actual_training_authorized':False}
        write(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':pin(out/'READY.json')}))
    except BaseException as error:
        write(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_models_optimizer_physics_GPU':0});raise
    finally:
        for module in components.values():module.close()


if __name__=='__main__':main()
