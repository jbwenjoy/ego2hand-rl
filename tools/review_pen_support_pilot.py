"""Fresh training3 restore, rejected-payload atomicity, and diagnostic continuation."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings
from supermimic_research.learning.native_pen_ppo import tensor_tree_equal
from review_owned_pen_state import compare_states


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    report=json.loads(a.training.read_text());cp=verify_file_pin(report['config']);cfg,ppo=pilot_settings(cp);out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh review directory')
    verify_source_snapshot(json.loads(verify_file_pin(report['source_receipt']).read_text()),live=True)
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/review_owned_pen_state.py',cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;updates=0;runtime=None;checks={};cases=[]
    # Retain the producer's qualified hook name, part of native4's callable contract.
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            for arm in report['arms']:
                name=arm['arm'];folder=out/name;folder.mkdir();state=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False)
                runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,name);audit=learner.load(copy.deepcopy(state));current=learner.checkpoint(receipt)
                local={k:v for k,v in audit.items() if isinstance(v,bool)}
                local.update({'native_'+k:v for k,v in compare_states(state['native'],current['native']).items()})
                keys=['modules','optimizer','memory','normalization_batch','observations','feature_sources','task_static','iteration','optimizer_steps','training_transitions',
                      'calibration_transitions','memory_iteration','rollout_complete','captured_rows','normalizers_ready']
                for key in keys:local['combined_'+key]=tensor_tree_equal(state[key],current[key])
                torch.save(current,folder/'fresh_state.pt')
                invalid=[]
                bad=copy.deepcopy(state);bad['schema']='pen_owned_contact_PPO_training:2';invalid.append(('old_schema',bad))
                if state['schema']=='pen_geometry_support_PPO_training:4':
                    bad=copy.deepcopy(state);bad['schema']='pen_geometry_support_PPO_training:3';invalid.append(('absolute_schema',bad))
                bad=copy.deepcopy(state);bad['metadata']['arm']='SUPPORT' if name=='RESIDUAL' else 'RESIDUAL';invalid.append(('cross_arm',bad))
                bad=copy.deepcopy(state)
                path=next(p for p in state['task_static']['tensors'] if state['native']['python']['tensors'][p].dtype==torch.float64)
                bad['native']['python']['tensors'][path].flatten()[0]+=1;invalid.append(('target_corruption',bad))
                for label,payload in invalid:
                    try:learner.load(payload)
                    except ValueError:local['reject_'+label]=True
                    else:local['reject_'+label]=False
                after=learner.checkpoint(receipt)
                local.update({'atomic_'+k:v for k,v in compare_states(current['native'],after['native']).items()})
                for key in keys:local['atomic_'+key]=tensor_tree_equal(current[key],after[key])
                # Resume the ACTUAL collector after Adam/norm/native state loading.
                # These four-control updates are diagnostics, never pilot candidates.
                before_native=observed;learner.collect(4);health=learner.consistency()
                local.update({'continued_'+k:v for k,v in health['checks'].items()})
                stats=learner.update();updates+=stats['optimizer_steps'];continued=learner.checkpoint(receipt)
                torch.save(continued,folder/'diagnostic_continuation.pt');again=learner.load(copy.deepcopy(continued))
                local.update({'continued_load_'+k:v for k,v in again.items() if isinstance(v,bool)})
                local['continued_native_bytes']=again['native']['all_array_bytes_equal'] and again['native']['all_scalar_values_equal']
                local['continued_budget']=observed-before_native==1024 and stats['optimizer_steps']==32
                local['continued_counters']=learner.iteration==state['iteration']+1 and learner.optimizer_steps==state['optimizer_steps']+32 and learner.training_transitions==state['training_transitions']+128
                expected_norm=32+state['calibration_transitions']+state['training_transitions']
                local['continued_norm_count']=int(learner.actor.norm.count)==int(learner.critic.norm.count)==expected_norm
                cases.append(dict(arm=name,checkpoint=arm['checkpoint'],fresh_state=file_pin(folder/'fresh_state.pt'),load_audit=audit,
                    diagnostic_continuation=file_pin(folder/'diagnostic_continuation.pt'),continuation_health=health,checks=local))
                checks.update({name+'_'+k:v for k,v in local.items()});verify_file_pin(arm['checkpoint']);runtime.close();runtime=None
            checks['observed_native_budget']=observed==2048;checks['diagnostic_update_budget']=updates==64
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='pen_geometry_support_pilot_state_review:1',training=file_pin(a.training),source_receipt=file_pin(out/'source_receipt.json'),
                cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,
                diagnostic_optimizer_steps=updates,retained_pilot_optimizer_steps=0,
                scope='Fresh combined training3/native4 readback, public schema/arm/target rejection atomicity and actual four-control loaded continuation/update/reload. Diagnostic updates excluded from pilot candidates; not W32 future determinism or task/ego success.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,diagnostic_updates=updates)),flush=True)
            if not all(checks.values()):raise ValueError('Retain state review discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),actual_native_integrations=observed,diagnostic_optimizer_steps=updates));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
