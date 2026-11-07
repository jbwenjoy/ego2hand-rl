"""Six narrow SYNTHETIC CPU cases; never load real data/checkpoint/native WM.

Snapshots current code before imports/execution. Native state mathematics and a
tiny surrogate exercise AD/copy/Adam/local checkpoint IO. This is not an actual
calibration run, an Env/native restore test or an accuracy qualification.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
DEX = Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic')
FILES = {'adapter': ROOT/'src/full151_gradient/response_calibration105_v1.py',
         'CLI': ROOT/'scripts/calibrate_full151_WM_common_pre_cpu_v1.py',
         'checker': Path(__file__),
         'independent_math': ROOT/'src/full151_gradient/common_pre_calibration_independent_math_v1.py',
         'native_state': DEX/'state.py', 'native_rotations': DEX/'rotations.py',
         'native_norm': DEX/'common.py'}


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/full151105_WM_calibration_preparation_CPU_v1')
    args=parser.parse_args(argv);out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir();before={k:pin(v) for k,v in FILES.items()}
    for k,v in FILES.items():
        (out/(k+'_source_before.py')).write_bytes(v.read_bytes())
    (out/'source_before.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf-8')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    sys.path.insert(0,str(ROOT/'src'))
    sys.path.insert(0,str(DEX.parents[2]))
    import numpy as np
    import torch
    from full151_gradient import response_calibration105_v1 as c
    from full151_gradient import common_pre_calibration_independent_math_v1 as ref
    from dex_supermimic.rl.dex_supermimic.state import difference, integrate
    from dex_supermimic.rl.dex_supermimic.common import RunningNorm
    torch.set_num_threads(4)
    checks={};details={};started=time.perf_counter()

    def reject(fn):
        try:fn()
        except ValueError:return True
        return False

    def state(R):
        a=np.zeros(58,np.float32);a[43:52]=np.asarray(R,dtype=np.float32).reshape(9);return a

    def rodrigues(v):
        v=np.asarray(v,dtype='f8');t=np.linalg.norm(v)
        if t==0:return np.eye(3)
        x,y,z=v/t;K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)

    # 1: Noncommuting rotations; compare native FP32 loss to independent F64
    # math with a stated fixture tolerance, never a physical arithmetic gate.
    preR=rodrigues([.4,.1,-.2]);pre_np=np.stack([state(preR)]*2)
    target_np=np.stack([state(preR@rodrigues([.2,.3,-.1])),state(preR@rodrigues([-.1,.4,.2]))])
    pred_np=np.stack([state(preR@rodrigues([.1,-.2,.3])),state(preR@rodrigues([.3,.2,-.2]))])
    target_np[:,:43]=np.linspace(-.2,.3,86,dtype='f4').reshape(2,43)
    pred_np[:,:43]=target_np[:,:43]+np.array([.02,-.04],np.float32)[:,None]
    pre=torch.from_numpy(pre_np);target=torch.from_numpy(target_np);pred=torch.from_numpy(pred_np)
    scale=torch.linspace(.5,2.,52);mi=torch.tensor([0]);pl=torch.tensor([1]);du=torch.tensor([.125],dtype=torch.float64)
    target_common=difference(pre,target).detach();diag={}
    lp,lr,_=c.common_pre_losses(pred,target,pre,scale,target_common,mi,pl,du,difference,torch,diagnostics=diag)
    controls=np.zeros((1,2,20),np.float32);controls[0,1,0]=.125
    independent=ref.calibration_losses(pred_np,target_np,pre_np[:1],pred_np[None],target_np[None],controls,np.array([0]),scale.numpy())
    wrong=(difference(pred[:1],pred[1:])-difference(target[:1],target[1:])).double()
    right=(diag['pred_common52_FP32'][1:]-diag['pred_common52_FP32'][:1]-(target_common[1:]-target_common[:1])).double()
    # Tolerance frozen before any real model/data; proper-rotation fixture only.
    checks['native_noncommuting_common_pre_matches_independent_math']=bool(
        abs(float(lp)-independent['Lpoint']) <= 2e-5*max(1.,independent['Lpoint'])
        and abs(float(lr)-independent['Lpair']) <= 2e-5*max(1.,independent['Lpair'])
        and float((wrong[...,43:46]-right[...,43:46]).abs().max()) > 1e-3)
    details['native_vs_independent_shadow']={'Lpoint_absdiff':abs(float(lp)-independent['Lpoint']),
        'Lpair_absdiff':abs(float(lr)-independent['Lpair']),'fixture_tolerance':2e-5,
        'not_native_FP32_bitwise_replacement':True}

    # 2: Actual native integrate/log backward through a synthetic parameter.
    delta=torch.zeros((2,52),requires_grad=True)
    prediction=integrate(pre,delta)
    lp,lr,_=c.common_pre_losses(prediction,target,pre,scale,target_common,mi,pl,du,difference,torch)
    (lp+lr).backward()
    checks['native_SO3_common_pre_synthetic_backward_finite']=bool(torch.isfinite(delta.grad).all() and delta.grad.abs().sum()>0)

    # 3: Any zero denominator blocks the complete batch; margin checks the
    # actual native rotvec norm, not a guessed trace/acos implementation.
    zero_rejected=reject(lambda:c.common_pre_losses(pred,target,pre,scale,target_common,mi,pl,torch.zeros_like(du),difference,torch))
    identity=torch.from_numpy(state(np.eye(3)))[None]
    close=torch.from_numpy(state(rodrigues([math.pi-.005,0,0])))[None]
    away=torch.from_numpy(state(rodrigues([math.pi-.02,0,0])))[None]
    c.chart_guard(identity,away,difference(identity,away),.01,torch)
    near_rejected=reject(lambda:c.chart_guard(identity,close,difference(identity,close),.01,torch))
    bad=difference(identity,away).clone();bad[0,0]=float('nan')
    checks['zero_actual_h_chart_cut_locus_nonfinite_stop_without_subset']=bool(zero_rejected and near_rejected and reject(lambda:c.chart_guard(identity,away,bad,.01,torch)))

    # 4: A small surrogate with the real RunningNorm/pack-integrate interfaces,
    # NOT native Networks.WorldModel and no real checkpoint/452/152 reads.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.norm=RunningNorm(298);self.delta_norm=RunningNorm(52)
            self.register_buffer('error_floor',torch.ones(52)*.01)
            self.net=torch.nn.Linear(78,67);self.amp=False
            with torch.no_grad():self.net.weight.fill_(.001);self.net.bias.zero_()
        def forward(self,x,u,dynamics):
            z=self.net(torch.cat((x,u),-1));return integrate(x,z[:,:52]*.01),z[:,52:]
    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__();self.wm=Tiny().eval().requires_grad_(False);self.register_buffer('physical_xi',torch.zeros(46))
        def forward(self,x,u):
            dynamics=torch.cat((x,self.physical_xi[None].expand(len(x),-1),x.new_zeros((len(x),174))),-1)
            return self.wm(x,u,dynamics)
    with torch.random.fork_rng(devices=[]):frozen=Frozen()
    a,b=c._make_trainable(frozen,torch),c._make_trainable(frozen,torch)
    before_parent=c._tensor_digest(frozen.wm.state_dict(),torch);before_b=c._tensor_digest(b.wm.state_dict(),torch)
    before_norm=c._tensor_digest(c._norm_state(a.wm),torch);before_cfg=c._norm_config(a.wm)
    tx=identity.expand(3,-1).clone();tu=torch.ones((3,20))*.1
    expected=frozen(tx,tu);ya=a(tx,tu);yb=b(tx,tu)
    parity=all(torch.equal(v.contiguous().view(torch.int32),w.contiguous().view(torch.int32)) for v,w in zip(expected,ya)) and all(torch.equal(v,w) for v,w in zip(expected,yb))
    optimizer=torch.optim.AdamW(a.wm.parameters(),lr=1e-4,weight_decay=0.,foreach=False)
    optimizer.zero_grad(set_to_none=True);ya[0].square().mean().backward();optimizer.step()
    checks['deepcopy_trainable_isolation_samebatch_parity_norm_xi_frozen']=bool(parity
        and all(not q.requires_grad for q in frozen.parameters())
        and all(q.data_ptr()!=r.data_ptr() for q,r in zip(a.wm.parameters(),frozen.wm.parameters()))
        and c._tensor_digest(frozen.wm.state_dict(),torch)==before_parent
        and c._tensor_digest(b.wm.state_dict(),torch)==before_b
        and c._tensor_digest(a.wm.state_dict(),torch)!=before_parent
        and c._tensor_digest(c._norm_state(a.wm),torch)==before_norm and c._norm_config(a.wm)==before_cfg
        and a.physical_xi.data_ptr()!=frozen.physical_xi.data_ptr() and torch.equal(a.physical_xi,frozen.physical_xi))

    # 5: A false external counter cannot replace actual Adam state.step.
    step=c.adam_step_evidence(optimizer,1,torch)
    wrong_step_rejected=reject(lambda:c.adam_step_evidence(optimizer,2,torch))
    payload={'schema':c.ARTIFACT_SCHEMA,'WM_state':a.wm.state_dict(),'optimizer_state':optimizer.state_dict(),
             'actual_Adam_step_evidence':step,'synthetic_only':True}
    testfile=out/'synthetic_component_checkpoint.pt';c._atomic_torch_save(testfile,payload,torch)
    restored=torch.load(testfile,map_location='cpu',weights_only=False)
    overwrite_rejected=reject(lambda:c._atomic_torch_save(testfile,payload,torch))
    checks['actual_Adam_step_and_atomic_checkpoint_readback']=bool(wrong_step_rejected and overwrite_rejected
        and c._tensor_digest(restored['WM_state'],torch)==c._tensor_digest(a.wm.state_dict(),torch)
        and restored['actual_Adam_step_evidence']['actual_step_values']==[1.]
        and all(float(v['step'])==1. for v in restored['optimizer_state']['state'].values())
        and not (out/'synthetic_component_checkpoint.pt.partial').exists())

    # 6: Only new CPU component authorization/namespace; AST top imports never
    # pull a backend on declaration. This is NOT an actual metadata-parent join.
    p={'schema':c.SCHEMA,'role':c.ROLE,'execution_authorized':False,'CPU_run_authorized':False,
       'GPU_run_authorized':False,'background_continue_to_final':True,'cpu_threads':4,
       'inputs':{k:{'path':'SYNTHETIC_NOT_READ','sha256':'0'*64} for k in c.INPUTS},
       'source_pins':{'synthetic':{'path':'SYNTHETIC_NOT_READ','sha256':'0'*64}},'recipe':copy.deepcopy(c.RECIPE),'output_prefix':c.OUTPUT_PREFIX}
    c.validate_protocol(p)
    unauthorized=reject(lambda:c.validate_protocol(p,require_authorized=True))
    good=copy.deepcopy(p);good['execution_authorized']=good['CPU_run_authorized']=True;c.validate_protocol(good,require_authorized=True)
    old=copy.deepcopy(p);old['schema']='full151_knownrough_native_training_run:2'
    badbool=copy.deepcopy(p);badbool['recipe']['iterations']=True
    tree=ast.parse(FILES['adapter'].read_text(encoding='utf-8'))
    imports=[ast.unparse(q) for q in tree.body if isinstance(q,(ast.Import,ast.ImportFrom))]
    checks['CPU_component_namespace_authorization_and_no_backend_top_imports']=bool(unauthorized
        and reject(lambda:c.validate_protocol(old)) and reject(lambda:c.validate_protocol(badbool))
        and not any(any(word in line for word in ('torch','numpy','mujoco','warp')) for line in imports))
    unchanged=all(pin(v)==before[k] for k,v in FILES.items())
    if not unchanged:raise RuntimeError('Executed source changed')
    if torch.cuda.is_initialized() or 'warp' in sys.modules or 'mujoco' in sys.modules:raise RuntimeError('Unexpected backend/GPU')
    report={'schema':'full151105_WM_common_pre_calibration_synthetic_CPU_contract:1',
            'status':'PASS_SYNTHETIC_NATIVE_MATH_COPY_ADAM_IO_ONLY' if all(checks.values()) else 'FAIL_SYNTHETIC_CONTRACT',
            'passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':before,'details':details,
            'scope':{'synthetic_Torch_CPU_AD_exercised':True,'synthetic_checkpoint_torch_loads':1,
                     'synthetic_surrogate_native_WM_architecture':False,
                     'actual_training_dataset_reads':0,'actual_parent_checkpoint_reads':0,'actual_native_WM_instances':0,
                     'actual_strict119f_loads':0,'actual_calibration_forwards_AD_optimizer_steps':0,
                     'actual_MJB_Env_physics_GPU':0},'source_unchanged':unchanged,'elapsed_seconds':time.perf_counter()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    ready={'schema':'full151105_WM_common_pre_calibration_synthetic_READY:1','status':report['status'],
           'source_pins':before,'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},
           'api':{'entry':'full151_gradient.response_calibration105_v1.run_calibration'},
           'actual_calibration_run_authorized':False}
    (out/'READY.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'READY':pin(out/'READY.json')}))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
