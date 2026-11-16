"""Actual CUDA-layout feature calculus,without changing simulator allocations."""
import argparse,copy,json,subprocess,sys
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh GPU feature-gradient check')
    out.mkdir();protocol=ROOT/'docs/pen-fixed-frame-gradient-protocol-v1.md';paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt);runtime=None;rows=[];checks={}
    try:
        with torch.jit.optimized_execution(False):
            runtime=SupportPenRuntime(ROOT/'configs/pen_centered_support_training_v1.toml',32);f=runtime.features;d=runtime.env.sim.data;g=copy.copy(f)
            original=f.motion.time_steps.clone();inputs=(d.qpos.clone(),d.qvel.clone(),f.action._raw_actions.clone(),f.action._processed_actions.clone())
            fake=SimpleNamespace(qpos=None,qvel=None,xpos=d.xpos,xquat=d.xquat)
            g.env=SimpleNamespace(sim=SimpleNamespace(data=fake,mj_model=runtime.env.sim.mj_model),scene=runtime.env.scene,device=runtime.device,num_envs=32)
            g.action=SimpleNamespace(_raw_actions=None,_processed_actions=None)
            def compute(mode,q,v,raw,EMA):
                g.fixed_frame_cache=mode;fake.qpos=q;fake.qvel=v;g.action._raw_actions=raw;g.action._processed_actions=EMA
                p=g.policy();return torch.cat([p,g.critic(p)],-1)
            for phase in [1,48,120,368]:
                f.motion.time_steps.fill_(phase)
                for pattern in [0,1]:
                    tangent=tuple((torch.linspace(-.02,.03,x.numel(),device=runtime.device).reshape_as(x) if pattern==0 else .025*torch.sin(torch.arange(x.numel(),device=runtime.device).reshape_as(x))) for x in inputs)
                    left,lj=torch.autograd.functional.jvp(lambda *x:compute(False,*x),inputs,tangent);right,rj=torch.autograd.functional.jvp(lambda *x:compute(True,*x),inputs,tangent)
                    error=float((left-right).abs().max());jerr=float((lj-rj).abs().max());name=f'P{phase}_D{pattern}'
                    valid=bool(torch.isfinite(left).all() and torch.isfinite(right).all() and torch.isfinite(lj).all() and torch.isfinite(rj).all()) and error<=2e-7 and jerr<=2e-6
                    rows.append(dict(name=name,forward_max_abs=error,JVP_max_abs=jerr,forward_exact=torch.equal(left,right),JVP_exact=torch.equal(lj,rj),passed=valid));checks[name]=valid
            f.motion.time_steps.copy_(original);checks['simulator_inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(inputs,(d.qpos,d.qvel,f.action._raw_actions,f.action._processed_actions)));runtime.validate_resources();runtime.reference_cache.validate(f);verify_source_snapshot(receipt,live=True)
            result=dict(schema='fixed_frame_dynamic_feature_GPU_JVP:1',protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
                scope='Numerical dynamic feature transformation derivatives with fixed mount/reference/phase,actual CUDA32 layout. Not physical contact/WM gradients,policy promotion or mobile-wrist qualification.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],max_JVP_error=max(r['JVP_max_abs'] for r in rows))),flush=True)
            if not all(checks.values()):raise ValueError('Retain feature-gradient discrepancy')
    finally:
        if runtime:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
