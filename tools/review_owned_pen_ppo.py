"""Fresh-process training2 state readback and independent fixed-data updates."""
import argparse
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.runtime.pen_portable_state import restore_rng
from supermimic_research.learning.owned_pen_ppo import OwnedPenPPO
from supermimic_research.learning.native_pen_ppo import cpu_tree,tensor_tree_equal
from review_native_pen_ppo import clone_training_modules
from review_owned_pen_state import compare_states


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    rp=a.report.resolve();run=json.loads(rp.read_text());cp=verify_file_pin(run['config']);cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['training_config']).resolve();train=tomllib.loads(tp.read_text());ap=(tp.parent/train['adapter_config']).resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total']:raise ValueError('Fresh healthy training2 review')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,tp,ap]
    paths += [ROOT/'tools'/n for n in ['review_native_pen_ppo.py','review_owned_pen_state.py','train_native_pen_ppo.py']]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    state=torch.load(verify_file_pin(run['checkpoint']),map_location='cpu',weights_only=False)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;clones=[];clone_steps=0
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            runtime=OwnedPenRuntime(ap,32);learner=OwnedPenPPO(runtime,train['ppo']);audit=learner.load(copy.deepcopy(state))
            checks={k:v for k,v in audit.items() if isinstance(v,bool)};checks.update(native_arrays=audit['native']['all_array_bytes_equal'],native_scalars=audit['native']['all_scalar_values_equal'])
            current=learner.checkpoint(receipt);torch.save(current,out/'fresh_combined_state.pt')
            checks.update({'native_'+k:v for k,v in compare_states(state['native'],current['native']).items()})
            for key in ['modules','optimizer','memory','normalization_batch','observations','feature_sources','iteration','optimizer_steps','training_transitions','calibration_transitions','memory_iteration','rollout_complete','captured_rows','normalizers_ready']:
                checks['combined_'+key]=tensor_tree_equal(state[key],current[key])
            source_before=cpu_tree(state);live_before=cpu_tree(current)
            memory={k:v.to(learner.device) for k,v in state['memory'].items()};final=state['observations']['critic'].to(learner.device)
            for _ in range(2):
                actor,critic,opt=clone_training_modules(state,learner.device);restore_rng(state['native']['rng'])
                stats=ppo_update(actor,critic,opt,memory,final,SimpleNamespace(**state['config']));clone_steps+=stats['optimizer_steps']
                clones.append(dict(actor=cpu_tree(actor.state_dict()),critic=cpu_tree(critic.state_dict()),optimizer=cpu_tree(opt.state_dict()),stats=cpu_tree(stats)))
            for key in clones[0]:checks['fixed_CUDA_'+key]=tensor_tree_equal(clones[0][key],clones[1][key])
            checks['source_modules_immutable']=tensor_tree_equal(source_before['modules'],state['modules']);checks['source_optimizer_immutable']=tensor_tree_equal(source_before['optimizer'],state['optimizer'])
            restore_rng(state['native']['rng']);after=learner.checkpoint(receipt)
            for key in ['modules','optimizer','memory','normalization_batch','observations']:checks['loaded_'+key+'_unchanged']=tensor_tree_equal(live_before[key],after[key])
            checks['load_fixed_updates_zero_physics']=observed==0;checks['clone_update_budget']=clone_steps==64
            # Exercise the actual loaded collector and next-block normalizer,
            # not just a copied network/fixed-data optimizer continuation.
            learner.collect(4);continuation_health=learner.consistency()
            checks.update({'continued_'+k:v for k,v in continuation_health['checks'].items()})
            continued_stats=learner.update();continued=learner.checkpoint(receipt);torch.save(continued,out/'continuation_state.pt')
            continued_audit=learner.load(copy.deepcopy(continued))
            checks.update({'continued_load_'+k:v for k,v in continued_audit.items() if isinstance(v,bool)})
            checks['continued_native_bytes']=continued_audit['native']['all_array_bytes_equal'];checks['continued_native_scalars']=continued_audit['native']['all_scalar_values_equal']
            checks['continued_counters']=learner.iteration==2 and learner.optimizer_steps==64 and learner.training_transitions==256
            checks['continued_normalization_count']=int(learner.actor.norm.count)==int(learner.critic.norm.count)==416
            checks['continued_physics_budget']=observed==1024
            verify_file_pin(run['checkpoint']);verify_source_snapshot(receipt,live=True)
            result=dict(schema='owned_pen_PPO_independent_review:1',qualification=file_pin(rp),checkpoint=run['checkpoint'],source_receipt=file_pin(out/'source_receipt.json'),
                fresh_state=file_pin(out/'fresh_combined_state.pt'),load_audit=audit,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,
                continuation_state=file_pin(out/'continuation_state.pt'),continuation_health=continuation_health,continuation_load_audit=continued_audit,
                diagnostic_clone_optimizer_steps=clone_steps,diagnostic_continuation_optimizer_steps=continued_stats['optimizer_steps'],retained_candidate_optimizer_steps=0,
                scope='Independent complete training2/native4 readback, two identical fixed-data CUDA optimizer clones and one fresh4-control loaded continuation/update/reload. Diagnostic tracking-only reward; no task/geometry/support/WM-gradient/ego, W32 future equality or learned-candidate claim.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,diagnostic_clone_steps=clone_steps,diagnostic_continued_steps=continued_stats['optimizer_steps'])))
            if not all(checks.values()):raise ValueError('Preserve independent training2 discrepancy')
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
