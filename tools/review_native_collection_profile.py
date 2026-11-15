"""Independent saved-rollout accounting and fixed-input reference feature cost."""
import argparse
import copy
import json
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from profile_native_collection import OperationCounts,timing
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.pen_reference_cache import FrozenPenReferenceCache
from supermimic_research.learning.native_pen_ppo import NativePenPPO
from supermimic_research.runtime.pen_portable_state import engine_sources,tensor_bytes


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    out=a.output.resolve();pr=a.profile.resolve();cr=a.cache.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh independent collection review')
    profile=json.loads(pr.read_text());cache_report=json.loads(cr.read_text());cp=verify_file_pin(cache_report['config']);cfg=tomllib.loads(cp.read_text())
    tp=(cp.parent/cfg['training_config']).resolve();train=tomllib.loads(tp.read_text());ap=(tp.parent/train['adapter_config']).resolve()
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/profile_native_collection.py',cp,tp,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};accounting=[]
    for name,r in [('profile',profile),('cache',cache_report)]:
        actions=torch.load(verify_file_pin(r['actions']),map_location='cpu',weights_only=False)
        for row in r['rows']:
            m=torch.load(verify_file_pin(row['memory']),map_location='cpu',weights_only=False);done=m['terminated']|m['truncated'];key=f"{name}_{row['mode']}_{row['repeat']}"
            local=dict(controls=m['actions'].shape==(64,32,20),same_forced_actions=torch.equal(m['actions'],actions),
                correct_applied=torch.equal(m['applied_actions'],m['actions'].clamp(-1,1)),
                phase_once=torch.equal(m['phase_next'],m['phase_source']+1),source_used=torch.equal(m['phase_source'],m['phase_used']),
                correct_reset_phase=torch.equal(m['phase_after_reset'],torch.where(done,torch.ones_like(m['phase_next']),m['phase_next'])),
                timeout_once=torch.equal(m['reward'],m['raw_reward']+train['ppo']['ppo_gamma']*m['terminal_next_value']*(m['truncated']&~m['terminated']).to(m['reward'].dtype)),
                no_force=bool(m['no_applied_force'].all()),no_overflow=not bool(m['engine_overflow'].any()),
                finite=all(bool(torch.isfinite(v).all()) for v in m.values() if v.is_floating_point()))
            checks.update({key+'_'+k:bool(v) for k,v in local.items()})
            accounting.append(dict(study=name,mode=row['mode'],repeat=row['repeat'],reset_calls=int(done.any(-1).sum()),reset_worlds=int(done.sum()),
                native_integrations=row['native_integrations'],collection_s=row.get('collection',row.get('timing'))['host_complete_s'],phase_max=int(m['phase_source'].max())))
    # Pure feature calls, same native tensors and phase, with no physics and no
    # old-policy parity audit inside the cached operation-count/timing region.
    from mjlab.sim.sim import Simulation
    original_step=Simulation.step;observed=0;adapter=None
    def step(sim):
        nonlocal observed
        original_step(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter(ap,32);trainer=NativePenPPO(adapter,train['ppo']);state=torch.load(verify_file_pin(cache_report['initial_state']),map_location='cpu',weights_only=False)
            audit=trainer.load(copy.deepcopy(state));f=adapter.features;cache=FrozenPenReferenceCache(f);cache.validate(f)
            before={k:tensor_bytes(v) for k,v in [('qpos',adapter.env.sim.data.qpos),('qvel',adapter.env.sim.data.qvel),('ctrl',adapter.env.sim.data.ctrl),('phase',f.motion.time_steps)]}
            micro={}
            for mode,fn in [('baseline',f.policy),('cached_reference',lambda:cache.policy(f))]:
                counter=OperationCounts()
                with counter:fn()
                for _ in range(128):fn()
                def batch():
                    for _ in range(512):fn()
                rows=[]
                for _ in range(3):_,duration=timing(batch);rows.append(duration)
                micro[mode]=dict(operation_count=counter.report(),measurements=rows,median_host_us_per_call=statistics.median(x['host_complete_s'] for x in rows)*1e6/512,
                    median_GPU_interval_us_per_call=statistics.median(x['GPU_stream_interval_s'] for x in rows)*1e6/512)
            after={k:tensor_bytes(v) for k,v in [('qpos',adapter.env.sim.data.qpos),('qvel',adapter.env.sim.data.qvel),('ctrl',adapter.env.sim.data.ctrl),('phase',f.motion.time_steps)]}
            checks.update(fixed_input_unchanged=before==after,fixed_input_feature_exact=torch.equal(f.policy(),cache.policy(f)),no_new_physics=observed==0,
                no_optimizer=trainer.optimizer_steps==0,initial_load=all(audit[k] for k in ['modules_equal','optimizer_equal','observation_equal','normalization_equal']))
            cache.validate(f)
    finally:
        Simulation.step=original_step
        if adapter:adapter.close()
    verify_source_snapshot(receipt,live=True)
    ratio=cache_report['medians']['cached_reference']['host_complete_s']/cache_report['medians']['baseline']['host_complete_s']
    result=dict(schema='native_collection_independent_cost_review:1',profile=file_pin(pr),cache=file_pin(cr),source_receipt=file_pin(out/'source_receipt.json'),engine_sources=engine_sources(),
        checks=checks,passed=sum(checks.values()),total=len(checks),accounting=accounting,fixed_input_micro=micro,observed_collection_time_reduction_fraction=1-ratio,
        native_integrations=observed,optimizer_steps=0,
        limits='Timed32-world rollouts have equal initial arrays/actions/native counts, but reset call counts differ and historical W32 replay remains unqualified. Cached live operation audit intentionally also calls the old encoder for parity; its raw counts are not a baseline-vs-cache comparison. Fixed-input counts here exclude that parity audit.',
        scope='Saved bookkeeping and same-input computational feature cost only; not new learned policy, task success, physical determinism, WM-gradient or training/ego speedup.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],fixed_feature_us={k:v['median_host_us_per_call'] for k,v in micro.items()},observed_collection_reduction=1-ratio)))
    if not all(checks.values()):raise ValueError('Preserve independent accounting discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
