"""Fresh GPU readback of all cached pilot checkpoints,zero new transitions."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    tp=a.training.resolve();run=json.loads(tp.read_text());cp=verify_file_pin(run['config']);cfg,ppo=pilot_settings(cp);out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg.get('fixed_frame_features')!='CACHED' or cfg['pilot_iterations']!=2 or run['passed']!=run['total']:raise ValueError('Fresh cached integration review required')
    q=json.loads(verify_file_pin(run['qualification']).read_text())
    if q['passed']!=q['total'] or q['actual_native_integrations']!=4096 or q['optimizer_steps']!=0:raise ValueError('Original cached qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()),live=True)
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,ROOT/'docs/pen-cached-adoption-protocol-v1.md']
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;cases=[];checks={}
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            for arm in run['arms']:
                runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,arm['arm'])
                for log in arm['logs']:
                    state=torch.load(verify_file_pin(log['checkpoint']),weights_only=False,map_location='cpu');audit=learner.load(state);label=arm['arm']+str(log['iteration'])
                    checks[label+'_native_bytes']=audit['native']['all_array_bytes_equal'] and audit['native']['all_scalar_values_equal']
                    checks[label+'_modules_Adam_obs_memory']=all(v for v in audit.values() if type(v) is bool)
                    checks[label+'_cached']=state['metadata']['fixed_frame_features']==runtime.metadata()['fixed_frame_features']=='CACHED'
                    checks[label+'_count']=int(learner.actor.norm.count)==32+learner.calibration_transitions+learner.training_transitions-64*32 and learner.normalization_seed_offset(state)==0
                    cases.append(dict(arm=arm['arm'],iteration=log['iteration'],checkpoint=log['checkpoint'],restore=audit,optimizer_steps=learner.optimizer_steps,normalizer_count=int(learner.actor.norm.count)))
                runtime.close();runtime=None
            checks.update(no_new_native_steps=observed==0,original_training_budget=run['actual_native_integrations']==65536 and run['optimizer_steps']==128)
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='cached_training_checkpoint_review:1',training=file_pin(tp),qualification=run['qualification'],source_receipt=file_pin(out/'source_receipt.json'),cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=0,scope='All four fresh CACHED training4 checkpoints reloaded into currentGPU32 runtimes,full native/module/Adam/observation/memory/normalizer readback. Not future trajectory determinism,policy promotion,task/ego or speed evidence.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],actual_native_integrations=observed)),flush=True)
            if not all(checks.values()):raise ValueError('Retain cached checkpoint review discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),actual_native_integrations=observed,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
