"""Probe projected initial EMA decay on the unchanged native action contract."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.runtime import owned_pen_state as owned
from supermimic_research.runtime.pen_portable_state import tensor_bytes


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['controls']!=4:raise ValueError('Fresh four-control EMA probe')
    out.mkdir();ap=(cp.parent/cfg['adapter_config']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;checks={};cases=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            runtime=OwnedPenRuntime(ap,1);source=owned.capture(runtime,receipt);torch.save(source,out/'source.pt');source_pin=file_pin(out/'source.pt')
            initial_EMA=runtime.initial_ema.clone();hold_raw=runtime.initial_raw.clone()
            checks['raw_holds_initial_ema_unclipped']=bool((hold_raw.abs()<=1).all()) and torch.equal(hold_raw,2*initial_EMA)
            for seed in cfg['noise_seeds']:
                generator=torch.Generator(device='cuda:0');generator.manual_seed(seed);eps=torch.randn(4,1,20,generator=generator,device='cuda:0')
                modes=[('ZERO_RAW',None),('HOLD_INITIAL_RAW',0.)]+[(f'HOLD_PLUS_NOISE_{std}',std) for std in cfg['noise_std']]
                for name,std in modes:
                    audit=owned.restore(runtime,source);checks['all_source_restores']=checks.get('all_source_restores',True) and audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                    rows=[];runtime.begin_collection();before=observed
                    for k in range(4):
                        raw=torch.zeros(1,20,device='cuda:0') if std is None else hold_raw[None]+std*eps[k]
                        EMA_before=runtime.action._processed_actions.clone();ob,reward,term,trunc,info=runtime.step(raw)
                        window={key:value.detach().cpu().clone() for key,value in info['contact_window'].items()}
                        EMA_after=runtime.action._processed_actions.clone();expected=.5*EMA_before+.25*raw.clamp(-1,1)
                        checks['EMA_recurrence']=checks.get('EMA_recurrence',True) and bool((EMA_after-expected).abs().max()<=1e-7)
                        if name=='HOLD_INITIAL_RAW':checks['hold_EMA_constant']=checks.get('hold_EMA_constant',True) and torch.equal(EMA_after[0],initial_EMA)
                        if name=='ZERO_RAW':checks['zero_EMA_halves']=checks.get('zero_EMA_halves',True) and torch.equal(EMA_after,.5*EMA_before)
                        rows.append(dict(control=k,used_phase=info['reference_phase'].cpu(),raw=raw.cpu(),applied=info['applied_raw'].cpu(),EMA_before=EMA_before.cpu(),EMA_after=EMA_after.cpu(),
                            PD_target=runtime.action._targets.cpu().clone(),ctrl=runtime.env.sim.wp_data.ctrl.numpy().copy(),window=window,
                            COM_error_m=info['COM_error_m'].cpu(),full_angle_rad=info['full_angle_rad'].cpu(),terminated=term.cpu(),truncated=trunc.cpu(),reward=reward.cpu(),
                            no_aid=not runtime.env.sim.wp_data.qfrc_applied.numpy().any() and not runtime.env.sim.wp_data.xfrc_applied.numpy().any()))
                        if bool((term|trunc).any()):break
                    runtime.end_collection();target=out/f'{name}_{seed}.pt';torch.save(rows,target)
                    g=torch.stack([r['window']['geometry_max'][0] for r in rows]);non=torch.stack([r['window']['nonhand_max_sum'][0,0] for r in rows])
                    gates=dict(object=bool((g[:,0]<=cfg['object_penetration_m']).all()),self=bool((g[:,1]<=cfg['self_penetration_m']).all()),joint=bool((g[:,2]<=cfg['joint_excursion_rad']).all()),
                        nonhand=bool((non<=cfg['nonhand_normal_N']).all()),finite_capacity=all(not r['window']['errors'].any() and not r['window']['global_error'].any() for r in rows),no_aid=all(r['no_aid'] for r in rows))
                    cases.append(dict(name=name,noise_seed=seed,noise_std=std,controls=len(rows),duration_s=.02*len(rows),raw_record=file_pin(target),native_integrations=observed-before,
                        object_pen_max_m=float(g[:,0].max()),self_pen_max_m=float(g[:,1].max()),joint_excursion_max_rad=float(g[:,2].max()),nonhand_max_N=float(non.max()),gates=gates,
                        geometry_prefix_qualified=all(gates.values()),full_task_qualified=False,final_EMA_delta_max=float((rows[-1]['EMA_after'][0]-initial_EMA.cpu()).abs().max())))
            checks['no_source_file_mutation']=file_pin(verify_file_pin(source_pin))==source_pin;checks['all_native_counts']=observed==sum(c['native_integrations'] for c in cases)
            checks['all_eight_samples']=all(c['native_integrations']==8*c['controls'] for c in cases);verify_source_snapshot(receipt,live=True)
            result=dict(schema='owned_initial_ema_feasibility_probe:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),source=source_pin,cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),
                initial_EMA_max_rad=float(initial_EMA.abs().max()),initial_raw_max=float(hold_raw.abs().max()),actual_native_integrations=observed,optimizer_steps=0,
                scope=cfg['scope']+' ZERO/HOLD repeats share exact actions/source: repeated diagnostics, not independent training seeds. Geometry-prefix pass is not COM/SO3/full7.36s task success. No reference retiming or applied force.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,
                cases=[dict(name=c['name'],seed=c['noise_seed'],object_mm=1000*c['object_pen_max_m'],self_mm=1000*c['self_pen_max_m'],geometry_ok=c['geometry_prefix_qualified']) for c in cases])),flush=True)
            if not all(checks.values()):raise ValueError('Preserve EMA/source probe discrepancy')
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
