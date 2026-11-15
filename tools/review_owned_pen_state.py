"""Independent saved owned-state bytes and fresh mixed-window partial restore."""
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
from supermimic_research.runtime import pen_portable_state as base


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def compare_states(a,b):
    checks={}
    for group in ['model','data']:
        x=a['native'][group];y=b['native'][group]
        checks[group+'_keys']=x['arrays'].keys()==y['arrays'].keys();checks[group+'_scalars']=x['scalars']==y['scalars']
        checks[group+'_bytes']=checks[group+'_keys'] and all(x['arrays'][k].tobytes()==y['arrays'][k].tobytes() for k in x['arrays'])
    for group in ['arrays','tensors']:
        x=a['python'][group];y=b['python'][group];checks['python_'+group+'_keys']=x.keys()==y.keys()
        def value(v):return base.tensor_bytes(v) if isinstance(v,torch.Tensor) else v.tobytes()
        checks['python_'+group+'_bytes']=checks['python_'+group+'_keys'] and all(value(v)==value(y[k]) for k,v in x.items())
    checks['python_scalars']=a['python']['scalars'].keys()==b['python']['scalars'].keys() and all(base.scalar_equal(v,b['python']['scalars'][k]) for k,v in a['python']['scalars'].items())
    checks['description']=a['python']['description']==b['python']['description'];checks['immutable']=all(a[k]==b[k] for k in ['immutable','immutable_arrays','immutable_scalars'])
    checks['rng_python']=a['rng']['python']==b['rng']['python'];checks['rng_numpy']=a['rng']['numpy'][0]==b['rng']['numpy'][0] and np.array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1]) and a['rng']['numpy'][2:]==b['rng']['numpy'][2:]
    checks['rng_Torch']=torch.equal(a['rng']['torch'],b['rng']['torch']) and len(a['rng']['cuda'])==len(b['rng']['cuda']) and all(torch.equal(x,y) for x,y in zip(a['rng']['cuda'],b['rng']['cuda']))
    return checks


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh independent owned-state review')
    out.mkdir();ap=(cp.parent/cfg['adapter_config']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/qualify_owned_pen_state.py',cp,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;cases=[];checks={};inputs=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            for W in [1,32]:
                pp=ROOT/f'runs/owned_pen_W{W}_producer_v6/report.json';rp=ROOT/f'runs/owned_pen_W{W}_restorer_v6/report.json'
                producer=json.loads(pp.read_text());restorer=json.loads(rp.read_text());inputs.extend([file_pin(pp),file_pin(rp)])
                s=torch.load(verify_file_pin(producer['checkpoint']),weights_only=False,map_location='cpu');t=torch.load(verify_file_pin(restorer['restored_state']),weights_only=False,map_location='cpu')
                compare=compare_states(s,t);checks.update({f'W{W}_mid_'+k:v for k,v in compare.items()})
                partial=torch.load(verify_file_pin(producer['partial_reset_state']),weights_only=False,map_location='cpu')
                runtime=OwnedPenRuntime(ap,W)
                try:
                    audit=owned.restore(runtime,partial);current=owned.capture(runtime,receipt);torch.save(current,out/f'W{W}_partial_restored.pt')
                    partial_checks=compare_states(partial,current);checks.update({f'W{W}_partial_'+k:v for k,v in partial_checks.items()})
                    checks[f'W{W}_partial_restore_audit']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                finally:runtime.close()
                one=torch.load(verify_file_pin(producer['future']),weights_only=False,map_location='cpu');two=torch.load(verify_file_pin(restorer['future']),weights_only=False,map_location='cpu')
                # Report future arithmetic WITHOUT promoting this one native tick
                # to a physical replay gate, whatever the observed difference.
                future={k:dict(exact=one[k].tobytes()==two[k].tobytes(),max_abs_difference=float(np.max(abs(one[k].astype(np.float64)-two[k])))) for k in ['qpos','qvel','ctrl','time']}
                future['WORLD_COM_wrench_max_abs_difference']=float(abs(one['contact']['wrench_WORLD_COM']-two['contact']['wrench_WORLD_COM']).max())
                future['same_contact_geom_order']=np.array_equal(one['contact']['geom'],two['contact']['geom']) and np.array_equal(one['contact']['worldid'],two['contact']['worldid'])
                cases.append(dict(worlds=W,producer=file_pin(pp),restorer=file_pin(rp),source_checkpoint=producer['checkpoint'],fresh_source=restorer['restored_state'],partial_source=producer['partial_reset_state'],
                    partial_restored=file_pin(out/f'W{W}_partial_restored.pt'),array_fields=restorer['audit']['array_fields'],scalar_fields=restorer['audit']['scalar_fields'],future_single_tick=future))
    finally:Simulation.step=original
    checks['zero_new_physics']=observed==0;verify_source_snapshot(receipt,live=True)
    result=dict(schema='owned_pen_state_independent_review:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),inputs=inputs,cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),
        actual_native_integrations=observed,optimizer_steps=0,scope='Independent complete saved native/Python/RNG byte accounting and fresh mixed-window partial restore. A one-tick future comparison is reported, not physical repeatability/WM-gradient/task/ego qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed)))
    if not all(checks.values()):raise ValueError('Preserve independent owned-state discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
