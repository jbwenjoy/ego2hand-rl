"""GPU timing/frame/reward/fork checks before any registered pilot update."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings,fork_initial_state
from supermimic_research.learning.native_pen_ppo import cpu_tree,tensor_tree_equal


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg,ppo=pilot_settings(cp);out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh preregistered pilot qualification')
    protocol=(cp.parent/cfg.get('protocol','../docs/geometry-support-pilot-protocol-v1.md')).resolve();selection=None
    if cfg.get('support_score','ABSOLUTE')=='CENTERED':
        sp=(cp.parent/cfg['score_study']).resolve();selection=json.loads(sp.read_text())
        if selection['selected_variant']!=['CENTERED'] or selection['passed']!=selection['total'] or selection['optimizer_steps'] or selection['actual_native_integrations']:raise ValueError('Passing preregistered offline score selection required')
        verify_source_snapshot(json.loads(verify_file_pin(selection['source_receipt']).read_text()),live=True)
        for pin in selection['inputs']:verify_file_pin(pin)
        for key in ['config','training','preview','prior_ledger']:verify_file_pin(selection[key])
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;checks={};cases=[];phase='construction'
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,'RESIDUAL');phase='common_calibration';learner.calibrate(8)
            common=learner.checkpoint(receipt);torch.save(common,out/'common_initial_state.pt');common_pin=file_pin(out/'common_initial_state.pt')
            table=runtime.support_table;np.savez_compressed(out/'support_reference.npz',targets=table.targets.cpu().numpy(),valid=table.valid.cpu().numpy(),pose=table.pose,mass=table.mass,inertia=table.inertia,gravity=table.gravity)
            checks['target_masks']=table.frames==369 and int(table.valid.sum())==365 and not bool(table.valid[-3:].any());runtime.close();runtime=None
            for arm in ['RESIDUAL','SUPPORT']:
                phase=arm;runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,arm);fork,changes=fork_initial_state(common,learner.metadata());audit=learner.load(fork)
                checks[arm+'_fork_arrays']=audit['native']['all_array_bytes_equal'] and audit['native']['all_scalar_values_equal']
                checks[arm+'_shared_modules']=tensor_tree_equal(common['modules'],fork['modules']);checks[arm+'_shared_observations']=tensor_tree_equal(common['observations'],fork['observations'])
                checks[arm+'_mean_initial_raw']=torch.equal(learner.actor.mean(learner.observations['policy']),runtime.initial_raw[None].expand(32,-1))
                learner.collect(4);health=learner.consistency();checks.update({arm+'_'+k:v for k,v in health['checks'].items()});m=cpu_tree(learner.memory);errors={}
                for row in range(4):
                    phase_id=m['phase_used'][row].numpy();idx=phase_id-1;world=np.arange(32);expected=table.targets.cpu().numpy()[world,idx]
                    R=m['contact_source_R'][row].double().numpy();target=np.c_[np.einsum('nji,nj->ni',R,expected[:,:3]),np.einsum('nji,nj->ni',R,expected[:,3:])]
                    actual=m['contact_objective_target_source'][row].numpy();errors['target']=max(errors.get('target',0.),float(abs(target-actual).max()))
                    measured=m['contact_wrench_sum'][row,:,5].numpy()/m['contact_counts'][row,:,0,None].numpy();errors['measured']=max(errors.get('measured',0.),float(abs(measured-m['contact_objective_measured_source'][row].numpy()).max()))
                    loss=((measured[:,:3]-target[:,:3])/cfg['force_scale_N'])**2;torque=((measured[:,3:]-target[:,3:])/cfg['torque_scale_Nm'])**2;score=np.exp(-.5*(loss.mean(-1)+torque.mean(-1)))
                    if cfg.get('support_score','ABSOLUTE')=='CENTERED':
                        baseline=np.exp(-.5*(((target[:,:3]/cfg['force_scale_N'])**2).mean(-1)+((target[:,3:]/cfg['torque_scale_Nm'])**2).mean(-1)))
                        score=score-baseline
                    eligible=m['contact_objective_support_eligible'][row].numpy();score=np.where(eligible,score,0).astype(np.float32)
                    errors['bonus']=max(errors.get('bonus',0.),float(abs(score-m['contact_objective_support_bonus'][row].numpy()).max()))
                    g=m['contact_geometry_max'][row].numpy();non=m['contact_nonhand_max_sum'][row,:,0].numpy();limits=np.array([.002,.0002,.001])
                    geom=(np.log1p((g/limits)**2).sum(-1)+np.log1p((non/.1)**2)).astype(np.float32)
                    reward=m['contact_objective_tracking_reward'][row].numpy()-np.float32(.25)*geom-np.float32(2)*m['contact_objective_geometry_failure'][row].numpy().astype(np.float32)+np.float32(cfg['support_weights'][arm])*score
                    errors['reward']=max(errors.get('reward',0.),float(abs(reward-m['raw_reward'][row].numpy()).max()))
                checks[arm+'_independent_target']=errors['target']<=1e-10;checks[arm+'_independent_measured']=errors['measured']<=1e-10;checks[arm+'_independent_bonus']=errors['bonus']<=1e-6;checks[arm+'_independent_reward']=errors['reward']<=1e-6
                target=out/f'{arm}_collector.pt';torch.save(m,target);state=learner.checkpoint(receipt);checks[arm+'_no_optimizer']=state['optimizer_steps']==0 and not state['optimizer']['state'];torch.save(state,out/f'{arm}_state.pt');reload=learner.load(copy.deepcopy(state))
                checks[arm+'_complete_reload']=reload['native']['all_array_bytes_equal'] and reload['native']['all_scalar_values_equal'] and all(v for v in reload.values() if isinstance(v,bool))
                cases.append(dict(arm=arm,health=health,load_audit=reload,fork_changes=changes,independent_max_errors=errors,memory=file_pin(target),state=file_pin(out/f'{arm}_state.pt')))
                runtime.close();runtime=None
            checks['common_file_unchanged']=file_pin(out/'common_initial_state.pt')==common_pin;checks['no_optimizer']=all(checks[arm+'_no_optimizer'] for arm in ['RESIDUAL','SUPPORT'])
            checks['actual_native_budget']=observed==4096;verify_source_snapshot(receipt,live=True)
            if selection is not None:checks['registered_score_selection']=selection['selected_variant']==['CENTERED']
            result=dict(schema='geometry_support_pilot_qualification:1',config=file_pin(cp),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),common_initial=common_pin,
                support_reference=file_pin(out/'support_reference.npz'),cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=0,
                scope='Preregistered reward/target/frame/fork/state qualification before pilot optimization. Nominal support is a reward only; no task/benefit/ego/physical WM-gradient claim.')
            if selection is not None:result['score_study']=file_pin(sp)
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed)),flush=True)
            if not all(checks.values()):raise ValueError('Preserve pre-pilot qualification discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
