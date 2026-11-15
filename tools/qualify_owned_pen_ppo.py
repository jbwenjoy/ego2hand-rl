"""GPU training2 collector, raw-contact rows and complete state integration."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.learning.owned_pen_ppo import OwnedPenPPO
from supermimic_research.learning.native_pen_ppo import NativePenPPO,cpu_tree,tensor_tree_equal
from supermimic_research.runtime.pen_resume_probe import substep_record
from qualify_native_pen_ppo import reference_end_source
from qualify_resident_contact_native import source_packet
from qualify_resident_contact_replay import oracle,empty_window,fold,canonical_joints


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def digest(state):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):h.update(str((x.shape,x.dtype)).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(state);return h.hexdigest()


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or (cfg['calibration_controls'],cfg['qualification_controls'],cfg['optimizer_updates'])!=(8,4,32):raise ValueError('Fresh declared training2 diagnostic')
    out.mkdir();tp=(cp.parent/cfg['training_config']).resolve();train=tomllib.loads(tp.read_text());ap=(tp.parent/train['adapter_config']).resolve()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,tp,ap]
    paths += [ROOT/'tools'/n for n in ['qualify_native_pen_ppo.py','qualify_resident_contact_native.py','qualify_resident_contact_replay.py']]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    lp=(cp.parent/cfg['prior_ledger']).resolve();ledger=json.loads(lp.read_text());aliases=json.loads(verify_file_pin(ledger['source_resolutions']).read_text());manifest=json.loads(verify_file_pin(ledger['verified_manifest']).read_text())
    for entry in manifest['records']:verify_source_pin(entry['observed'],aliases)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;control=OwnedPenRuntime.step;observed=0;updates=0;runtime=None;record=False;packets=[];starts=[];checks={};phase='construction';rejections=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
        if record and runtime is not None and sim is runtime.env.sim:
            p=substep_record(runtime);p['cone']=int(sim.wp_model.opt.cone);packets.append(p)
    def control_step(self,raw):
        if record and self is runtime:starts.append(source_packet(self))
        return control(self,raw)
    Simulation.step=step;OwnedPenRuntime.step=control_step
    try:
        with torch.jit.optimized_execution(False):
            runtime=OwnedPenRuntime(ap,32);trainer=OwnedPenPPO(runtime,train['ppo']);phase='calibration';trainer.calibrate(8)
            phase='collection';record=True;trainer.collect(4);record=False;health=trainer.consistency();checks.update({'collector_'+k:v for k,v in health['checks'].items()})
            torch.save(packets,out/'native_packets.pt');torch.save(starts,out/'window_sources.pt');torch.save(cpu_tree(trainer.memory),out/'collector_memory.pt')
            s=runtime.env.sim;maximum={};raw_checks={};bounds=s.wp_model.jnt_range.numpy()
            for row,start in enumerate(starts):
                window=empty_window(start)
                for p in packets[row*8:row*8+8]:expected,_,_=oracle(s.mj_model,runtime.features.body,canonical_joints(s.mj_model),bounds,p);fold(window,expected,p)
                for key,value in window.items():
                    actual=trainer.memory['contact_'+key][row].cpu().numpy();error=float(np.max(abs(actual.astype(np.float64)-value))) if actual.size else 0.
                    maximum[key]=max(maximum.get(key,0.),error);raw_checks[key]=raw_checks.get(key,True) and bool(np.allclose(actual,value,atol=1e-9,rtol=1e-12))
            checks.update({'raw_window_'+k:v for k,v in raw_checks.items()});checks['raw_native_coverage']=len(packets)==32 and len(starts)==4
            pre=trainer.checkpoint(receipt);torch.save(pre,out/'preupdate_state.pt')
            phase='diagnostic_update';stats=trainer.update();updates+=stats['optimizer_steps'];checks['nonempty_optimizer']=bool(trainer.optimizer.state_dict()['state'])
            state=trainer.checkpoint(receipt);torch.save(state,out/'checkpoint.pt');checkpoint_pin=file_pin(out/'checkpoint.pt');checks['update_budget']=trainer.iteration==1 and trainer.optimizer_steps==32
            checks['parameters_changed']=any(not torch.equal(pre['modules'][name][key],value) for name in ['actor','critic'] for key,value in state['modules'][name].items() if key in dict(getattr(trainer,name).named_parameters()))
            def reject(name,bad):
                before=digest(trainer.checkpoint(receipt));reason=None
                try:trainer.load(bad)
                except (ValueError,KeyError,TypeError,RuntimeError) as error:reason=str(error)
                after=digest(trainer.checkpoint(receipt));checks[name+'_rejected']=reason is not None;checks[name+'_unchanged']=before==after
                rejections.append(dict(name=name,reason=reason,before=before,after=after))
            phase='negative_loads';bad=copy.deepcopy(state);bad['schema']='pen_native_full_pose_PPO_training:1';reject('old_schema',bad)
            bad=copy.deepcopy(state);bad['memory']['contact_counts'][0,0,0]=7;reject('contact_count',bad)
            bad=copy.deepcopy(state);bad['normalization_batch']['policy'][0,0]+=1;reject('norm_batch',bad)
            bad=copy.deepcopy(state);bad['observations']['policy'][0,132]+=.01;reject('observation_source',bad)
            bad=copy.deepcopy(state);first=next(iter(bad['optimizer']['state']));bad['optimizer']['state'][first]['step']=torch.tensor([1.]);reject('optimizer_step',bad)
            checks['source_file_immutable']=file_pin(out/'checkpoint.pt')==checkpoint_pin
            checks['no_hook_left']=not trainer._collecting and 'step' not in runtime.__dict__ and not runtime._collection_open
            primary_health=health;primary_metadata=trainer.metadata();runtime.close();runtime=None
            phase='fresh_combined_load';runtime=OwnedPenRuntime(ap,32);loaded=OwnedPenPPO(runtime,train['ppo']);before=observed;load_audit=loaded.load(copy.deepcopy(state))
            checks.update({'load_'+k:v for k,v in load_audit.items() if isinstance(v,bool)});checks['load_native_bytes']=load_audit['native']['all_array_bytes_equal'];checks['load_native_scalars']=load_audit['native']['all_scalar_values_equal'];checks['load_no_physics']=observed==before
            torch.save(loaded.checkpoint(receipt),out/'restored_state.pt');runtime.close();runtime=None
            phase='timeout_prestart';runtime=OwnedPenRuntime(ap,1);reference_end_source(runtime);last=OwnedPenPPO(runtime,train['ppo']);last.collect(1,forced_actions=torch.zeros(1,1,20,device='cuda:0'))
            timeout_health=last.consistency();checks.update({'timeout_'+k:v for k,v in timeout_health['checks'].items()})
            checks['timeout_real_truncation']=bool(last.memory['truncated'][0,0]) and not bool(last.memory['terminated'][0,0])
            checks['timeout_contact_retained']=int(last.memory['contact_counts'][0,0,0])==8 and int(runtime.contact_views['counts'][0,0])==0
            torch.save(cpu_tree(last.memory),out/'timeout_memory.pt');checks['actual_native_budget']=observed==3080;checks['actual_optimizer_budget']=updates==32
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='owned_pen_PPO_integration_qualification:1',config=file_pin(cp),prior_evidence=file_pin(lp),source_receipt=file_pin(out/'source_receipt.json'),checkpoint=file_pin(out/'checkpoint.pt'),
                preupdate_state=file_pin(out/'preupdate_state.pt'),restored_state=file_pin(out/'restored_state.pt'),memory=file_pin(out/'collector_memory.pt'),packets=file_pin(out/'native_packets.pt'),window_sources=file_pin(out/'window_sources.pt'),timeout_memory=file_pin(out/'timeout_memory.pt'),
                primary_health=primary_health,timeout_health=timeout_health,metadata=primary_metadata,load_audit=load_audit,raw_max_errors=maximum,rejections=rejections,
                checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,diagnostic_optimizer_steps=updates,retained_candidate_optimizer_steps=0,
                scope=cfg['scope']+' Raw native packets validate arithmetic only; the shared PPO update ignores contact columns. Calibration costs included; no task-success/geometry reward/CHORD/WM-gradient/ego or W32 physical-future qualification.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,diagnostic_updates=updates)),flush=True)
            if not all(checks.values()):raise ValueError('Keep training2 integration discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,diagnostic_optimizer_steps=updates));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original;OwnedPenRuntime.step=control


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
