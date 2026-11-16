"""Actual full runtime/graph/control/state checks for explicit collision backend."""
import argparse
import copy
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.learning.backend_pen_ppo import BackendSupportPenPPO
from supermimic_research.learning.pen_support_ppo import pilot_settings
from supermimic_research.learning.pen_matched_recovery import WorkCounter,saved_bytes_equal,persist_failed_learner,write_metadata as write
from supermimic_research.learning.native_pen_ppo import cpu_tree
from supermimic_research.mechanics.portable_contact_records import contacts
from qualify_resident_contact_native import source_packet
from qualify_resident_contact_replay import oracle,empty_window,fold,canonical_joints
from train_native_pen_ppo import finite_training_state


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['worlds'],cfg['arms'],cfg['calibration_controls'],cfg['oracle_controls'],cfg['training_controls'],cfg['post_resume_controls'],cfg['max_native_integrations'],cfg['max_optimizer_calls'])!=('EXPLICIT_COLLISION_BACKEND_INTEGRATION',[1,32],['RESIDUAL','SUPPORT'],8,4,64,4,42240,128):raise ValueError('Fixed four-case integration budget required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['objective_config','backend_manifest','protocol','registration']}
    contract=load_contract(paths['backend_manifest']);inputs={k:file_pin(paths[k]) for k in ['objective_config','backend_manifest','protocol']}
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['objective_config'],paths['backend_manifest'],paths['protocol']]+[ROOT/'tools'/n for n in ['qualify_resident_contact_native.py','qualify_resident_contact_replay.py','train_native_pen_ppo.py']]
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='explicit_backend_integration_registration:1',config=file_pin(cp),inputs=inputs,backend_id=contract['backend_id'],source_receipt=file_pin(out/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['backend_id']!=contract['backend_id']:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh integration output required')
    out.mkdir();receipt=freeze(out)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;counter=WorkCounter();runtime=learner=None;record=False;packets=[];starts=[];sample=0;phase='backend_scope';folder=out
    def step(sim):
        nonlocal sample
        observing=record and runtime is not None and sim is runtime.env.sim
        if observing and sample%8==0:starts.append(source_packet(runtime))
        result=counter.native_call(original,sim)
        if observing:
            d=sim.wp_data;c=contacts(sim.wp_model,d,sim.mj_model,runtime.features.body);qa=int(sim.mj_model.jnt_dofadr[sim.mj_model.body_jntadr[runtime.features.body]])
            native=d.qfrc_constraint.numpy()[:,qa:qa+6].copy();world=np.concatenate((native[:,:3],np.einsum('wij,wj->wi',c['R_efc_phase_world'],native[:,3:])),axis=-1)
            packets.append(dict(qpos=d.qpos.numpy().copy(),time=d.time.numpy().copy(),cone=int(sim.wp_model.opt.cone),contact=c,constraint_wrench_WORLD=world,
                no_applied_force=not bool(np.any(d.qfrc_applied.numpy()) or np.any(d.xfrc_applied.numpy()))))
            sample+=1
        return result
    Simulation.step=step;checks={};cases=[];backend=None
    try:
        torch.set_num_threads(4);_,ppo=pilot_settings(paths['objective_config'])
        with backend_scope(paths['backend_manifest']) as backend,torch.jit.optimized_execution(False):
            for W in cfg['worlds']:
                for arm in cfg['arms']:
                    label=f'W{W}_{arm}';folder=out/label;folder.mkdir();phase=label+'_construction';before=counter.native;build_before=len(backend.builds)
                    runtime=BackendSupportPenRuntime(paths['objective_config'],W);learner=BackendSupportPenPPO(runtime,ppo,arm)
                    local={};maximum={};local['fresh_constructor_no_native_steps']=counter.native==before
                    local['candidate_kernels_captured']=len(backend.builds)>build_before and runtime.env.sim.step_graph is not None and runtime.env.sim.forward_graph is not None
                    write(folder/'construction_backend_audit.json',backend.audit())
                    phase=label+'_calibration';learner.calibrate(8);counter.bind_optimizer(learner.optimizer)
                    phase=label+'_raw_oracle';packets=[];starts=[];sample=0;record=True
                    try:learner.collect(4)
                    finally:record=False
                    health=learner.consistency();write(folder/'oracle_health.json',health);local.update({'oracle_'+k:v for k,v in health['checks'].items()})
                    torch.save(dict(packets=packets,starts=starts,memory=cpu_tree(learner.memory)),folder/'raw_oracle.pt')
                    s=runtime.env.sim;bounds=s.wp_model.jnt_range.numpy();jids=canonical_joints(s.mj_model)
                    for row,start in enumerate(starts):
                        window=empty_window(start)
                        for packet in packets[row*8:row*8+8]:
                            expected,_,_=oracle(s.mj_model,runtime.features.body,jids,bounds,packet);fold(window,expected,packet)
                            local['raw_contacts_valid']=local.get('raw_contacts_valid',True) and all(packet['contact']['flags'].values()) and packet['no_applied_force']
                            delta=np.abs(packet['constraint_wrench_WORLD']-packet['contact']['wrench_WORLD_COM'])
                            maximum['constraint_force']=max(maximum.get('constraint_force',0.),float(delta[:,:3].max()));maximum['constraint_torque']=max(maximum.get('constraint_torque',0.),float(delta[:,3:].max()))
                        for key,value in window.items():
                            actual=learner.memory['contact_'+key][row].cpu().numpy();err=float(np.max(abs(actual.astype(np.float64)-value))) if actual.size else 0.
                            maximum['raw_'+key]=max(maximum.get('raw_'+key,0.),err)
                            local['raw_'+key]=local.get('raw_'+key,True) and bool(np.allclose(actual,value,atol=cfg['raw_absolute_tolerance'],rtol=cfg['raw_relative_tolerance']))
                    local['oracle_coverage']=len(packets)==32 and len(starts)==4
                    local['constraint_force_balance']=maximum['constraint_force']<=cfg['constraint_force_tolerance_N'];local['constraint_torque_balance']=maximum['constraint_torque']<=cfg['constraint_torque_tolerance_Nm']
                    phase=label+'_partial_reset';selected=[0] if W==1 else [0,7,31];ids=torch.tensor(selected,device=runtime.device);keep=np.setdiff1d(np.arange(W),selected)
                    pre={k:getattr(s.wp_data,k).numpy().copy() for k in ['qpos','qvel','ctrl','time']};learner.observations=runtime.reset(ids)
                    local['unselected_physical_state_unchanged']=all(np.array_equal(getattr(s.wp_data,k).numpy()[keep],v[keep]) for k,v in pre.items())
                    local['selected_phase_reset']=bool((runtime.motion.time_steps[ids]==1).all())
                    phase=label+'_collect64';learner.collect(64);health=learner.consistency();write(folder/'training_health.json',health);local.update({'training_'+k:v for k,v in health['checks'].items()})
                    write(folder/'preupdate_checks.json',dict(checks=local,maximum=maximum))
                    if not all(local.values()):raise ValueError('Integration check failed before optimizer: '+repr([k for k,v in local.items() if not v]))
                    phase=label+'_one_update';stats=learner.update();local['finite_Adam_update']=stats['optimizer_steps']==32 and finite_training_state(learner)
                    phase=label+'_checkpoint';state=learner.checkpoint(receipt);torch.save(state,folder/'checkpoint_0001.pt')
                    loaded=torch.load(folder/'checkpoint_0001.pt',map_location='cpu',weights_only=False);local['full_CPU_saved_bytes']=saved_bytes_equal(state,loaded)
                    if W==1 and arm=='RESIDUAL':
                        phase='negative_identity_and_binding_tests';rejects=[]
                        def reject_state(name,bad):
                            before_state=learner.checkpoint(receipt);n,u=counter.native,counter.optimizer;reason=None
                            try:learner.load(bad)
                            except (ValueError,KeyError,TypeError) as error:reason=str(error)
                            after_state=learner.checkpoint(receipt)
                            local[name]=reason is not None and counter.native==n and counter.optimizer==u and saved_bytes_equal(before_state,after_state)
                            rejects.append(dict(name=name,reason=reason,no_partial_write=local[name]))
                        old=torch.load(ROOT/'runs/pen_long_control_training_v1/RESIDUAL/checkpoint_0200.pt',map_location='cpu',weights_only=False);reject_state('old_backend_rejected',old)
                        bad=copy.deepcopy(state);p=(('attr','_collision_backend_contract'),('item','backend_id'));bad['native']['python']['scalars'][p]='corrupt';reject_state('owned_backend_corruption_rejected',bad)
                        for name,target,attr,replacement in [('entry',backend.driver,'convex_narrowphase',backend.installed),('factory',backend.convex,'ccd_kernel_builder_face_normal_v1',lambda *a:None)]:
                            good=getattr(target,attr);n=counter.native;reason=None
                            try:
                                setattr(target,attr,replacement)
                                try:runtime.step(torch.zeros(W,20,device=runtime.device))
                                except ValueError as error:reason=str(error)
                            finally:setattr(target,attr,good)
                            local[name+'_tamper_no_physics']=reason is not None and n==counter.native;rejects.append(dict(name=name,reason=reason,no_native=local[name+'_tamper_no_physics']))
                        write(folder/'rejections.json',rejects)
                    runtime.close();runtime=None;phase=label+'_fresh_resume';runtime=BackendSupportPenRuntime(paths['objective_config'],W);learner=BackendSupportPenPPO(runtime,ppo,arm)
                    n=counter.native;audit=learner.load(loaded);local['resume_no_native_steps']=counter.native==n
                    local.update({'load_'+k:v for k,v in audit.items() if type(v) is bool});local['load_all_native_arrays']=audit['native']['all_array_bytes_equal'];local['load_all_native_scalars']=audit['native']['all_scalar_values_equal']
                    write(folder/'reload.json',audit);phase=label+'_resume_collect';learner.collect(4);health=learner.consistency();write(folder/'resume_health.json',health);local.update({'resume_'+k:v for k,v in health['checks'].items()})
                    local['resume_has_nonempty_Adam']=learner.optimizer_steps==32 and bool(learner.optimizer.state_dict()['state'])
                    write(folder/'checks.json',dict(checks=local,passed=sum(local.values()),total=len(local),maximum=maximum))
                    if not all(local.values()):raise ValueError('Backend case qualification failed: '+repr([k for k,v in local.items() if not v]))
                    cases.append(dict(label=label,checks=file_pin(folder/'checks.json'),checkpoint=file_pin(folder/'checkpoint_0001.pt'),reload=file_pin(folder/'reload.json'),
                        raw_oracle=file_pin(folder/'raw_oracle.pt'),native_integrations=counter.native-before,optimizer_steps=32))
                    print(json.dumps(dict(case=label,checks=[sum(local.values()),len(local)],**counter.metadata())),flush=True);runtime.close();runtime=learner=None
            audit=backend.audit();write(out/'backend_audit.json',audit)
        checks=dict(four_fresh_cases=len(cases)==4,actual_native_budget=counter.native==42240,actual_Adam_budget=counter.optimizer==128,
            no_partial_calls=counter.metadata()['completed_call_accounting_exact'],all_runtimes_closed=audit['live_runtimes']==0)
        from mujoco_warp._src import collision_driver,collision_convex
        checks['installed_binding_restored']=collision_driver.convex_narrowphase is collision_convex.convex_narrowphase
        verify_source_snapshot(receipt,live=True)
        result=dict(schema='explicit_collision_backend_integration:1',config=file_pin(cp),registration=file_pin(paths['registration']),source_receipt=file_pin(out/'source_receipt.json'),backend_manifest=inputs['backend_manifest'],
            cases=cases,backend_audit=file_pin(out/'backend_audit.json'),checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),wall_s=time.perf_counter()-started,
            scope='Full fresh runtime/graph/raw-wrench/partial-reset/new-backend checkpoint diagnostic. No old-state migration,final task/WM-gradient/ego or speedup admission.')
        write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],**counter.metadata())),flush=True)
        if not all(checks.values()):raise ValueError('Backend integration budget/binding check failed')
    except Exception as error:
        write(out/'failure_exception.json',dict(type=type(error).__name__,message=str(error),phase=phase,**counter.metadata()))
        # Keep the backend active until available runtime evidence is captured.
        evidence={}
        try:
            if learner is not None:evidence=persist_failed_learner(learner,folder,receipt)
        except Exception as e:evidence=dict(capture_error=str(e))
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,evidence=evidence,**counter.metadata(),wall_s=time.perf_counter()-started));raise
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception:pass
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
