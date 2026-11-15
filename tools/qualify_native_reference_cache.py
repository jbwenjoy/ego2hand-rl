"""Same-input full-phase/live feature parity and matched collection timing."""
import argparse
import copy
import json
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from profile_native_collection import OperationCounts,timing
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter,NativePenFeatures
from supermimic_research.runtime.pen_reference_cache import FrozenPenReferenceCache
from supermimic_research.runtime.pen_portable_state import engine_sources
from supermimic_research.learning.native_pen_ppo import NativePenPPO,cpu_tree


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or (cfg['num_envs'],cfg['controls'],cfg['repeats'])!=(32,64,3):raise ValueError('Fresh declared reference cache protocol')
    out.mkdir();tp=(cp.parent/cfg['training_config']).resolve();train=tomllib.loads(tp.read_text());ap=(tp.parent/train['adapter_config']).resolve()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/profile_native_collection.py',cp,tp,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    lp=(cp.parent/cfg['prior_ledger']).resolve();ledger=json.loads(lp.read_text());first=json.loads(verify_file_pin(ledger['roots']['first2']).read_text());pin=first['initial_state']
    source=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
    from mjlab.sim.sim import Simulation
    oldstep=Simulation.step;oldpolicy=NativePenFeatures.policy;adapter=None;cache=None;use_cache=False;audit=False;differences=[];observed=0;phase='construction'
    def step(sim):
        nonlocal observed
        oldstep(sim);observed+=sim.num_envs
    def policy(features):
        if not use_cache:return oldpolicy(features)
        cached=cache.policy(features)
        if audit:
            original=oldpolicy(features);differences.append((original-cached).abs().max())
        return cached
    Simulation.step=step;NativePenFeatures.policy=policy
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter(ap,32);trainer=NativePenPPO(adapter,train['ppo']);f=adapter.features
            def restore():
                r=trainer.load(copy.deepcopy(source))
                if not all(r[k] for k in ['modules_equal','optimizer_equal','observation_equal','normalization_equal']) or not r['native']['all_array_bytes_equal'] or not r['native']['all_scalar_values_equal']:raise ValueError('Initial state load mismatch')
            restore();cache=FrozenPenReferenceCache(f);cache.validate(f);parity=[]
            # Every phase and clamp boundary; use the same current native state
            # for both encoders. No integration or qpos teleport is needed.
            phase='full_phase_parity'
            for t in [-3,0,*range(1,369),369,400]:
                f.motion.time_steps.fill_(t);original=oldpolicy(f);cached=cache.policy(f)
                parity.append(dict(phase=t,max_error=float((original-cached).abs().max()),exact=torch.equal(original,cached)))
            f.motion.time_steps.copy_(torch.arange(32,device=adapter.device)*11)
            mixed_error=float((oldpolicy(f)-cache.policy(f)).abs().max())
            restore();w=torch.arange(32,device=adapter.device)[None,:,None];j=torch.arange(20,device=adapter.device)[None,None,:];t=torch.arange(64,device=adapter.device)[:,None,None]
            actions=.05*torch.sin(.2*t+.13*w+.03*j);torch.save(actions.cpu(),out/'forced_actions.pt')
            phase='warmup';trainer.collect(64,forced_actions=actions);rows=[];counts={}
            for mode in ['baseline','cached_reference']:
                use_cache=mode=='cached_reference'
                for repeat in range(3):
                    phase=f'{mode}_{repeat}';restore();cache.validate(f);before=observed
                    _,duration=timing(lambda:trainer.collect(64,forced_actions=actions));health=trainer.consistency();cache.validate(f)
                    memory=out/f'{mode}_{repeat}_memory.pt';torch.save(cpu_tree(trainer.memory),memory)
                    rows.append(dict(mode=mode,repeat=repeat,timing=duration,consistency=health,memory=file_pin(memory),native_integrations=observed-before))
                    write(out/'progress.json',dict(phase=phase,observed_native_integrations=observed,optimizer_steps=0,rows=rows))
                phase=f'{mode}_operation_audit';restore();counter=OperationCounts();audit=use_cache
                with counter:trainer.collect(64,forced_actions=actions)
                audit=False;health=trainer.consistency();counts[mode]=dict(**counter.report(),consistency=health)
            max_live=float(torch.stack(differences).max());cache.validate(f)
            checks=dict(full_phase_parity=all(x['max_error']<=cfg['absolute_tolerance'] for x in parity),mixed_phase_parity=mixed_error<=cfg['absolute_tolerance'],
                live_same_input_parity=max_live<=cfg['absolute_tolerance'],all_collection_consistent=all(r['consistency']['passed']==r['consistency']['total'] for r in rows),
                operation_audits_consistent=all(c['consistency']['passed']==c['consistency']['total'] for c in counts.values()),native_budget=observed==147456,
                per_rollout_budget=all(r['native_integrations']==16384 for r in rows),no_optimizer=trainer.optimizer_steps==source['optimizer_steps']==0,
                source_unchanged=file_pin(verify_file_pin(pin))==pin)
            write(out/'operation_counts.json',counts);write(out/'full_phase_parity.json',dict(cases=parity,mixed_error=mixed_error,live_same_input_calls=len(differences),live_max_error=max_live))
            medians={mode:{k:statistics.median(r['timing'][k] for r in rows if r['mode']==mode) for k in rows[0]['timing']} for mode in counts}
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='native_reference_cache_qualification:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),initial_state=pin,engine_sources=engine_sources(),
                actions=file_pin(out/'forced_actions.pt'),cache=cache.metadata(),rows=rows,medians=medians,checks=checks,passed=sum(checks.values()),total=len(checks),
                operation_counts=file_pin(out/'operation_counts.json'),parity=file_pin(out/'full_phase_parity.json'),actual_native_integrations=observed,optimizer_steps=0,
                scope=cfg['scope']+' Per-call old/new parity audits share identical current state. Timed baseline/cache trajectories can have W32 native numerical variation; timings alone do not establish physics determinism or learned performance.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],medians=medians,max_live_error=max_live,native_integrations=observed)))
            if not all(checks.values()):raise ValueError('Retain reference cache discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,observed_native_integrations=observed,optimizer_steps=0));raise
    finally:
        Simulation.step=oldstep;NativePenFeatures.policy=oldpolicy
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
