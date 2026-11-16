"""Fresh same-source backend study and pre-registered strict final execution."""
import argparse
import copy
import json
import subprocess
import sys
import time
import tomllib
from functools import partial
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract,active_backend
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.learning.backend_pen_ppo import BackendSupportPenPPO
from supermimic_research.learning.backend_control_study import study_budget,require_fresh_common,require_final_training
from supermimic_research.learning.pen_support_ppo import pilot_settings,fork_initial_state
from supermimic_research.learning.pen_matched_recovery import WorkCounter,saved_bytes_equal,persist_failed_learner,write_metadata as write
from supermimic_research.learning.native_pen_ppo import cpu_tree,tensor_tree_equal
from supermimic_research.learning.pen_w1_evaluation import actor_for_fresh_live_W1
from supermimic_research.learning.gpu_monitor import GPUMonitor
from supermimic_research.runtime.pen_executed_control_audit import audit_executed_control
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from train_native_pen_ppo import finite_training_state
from evaluate_pen_long_control import context_pins
import preview_pen_support_pilot as preview


def save(learner,receipt,path):
    if path.exists():raise ValueError('No checkpoint replacement')
    state=learner.checkpoint(receipt);temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
    return file_pin(path),state


def actor_initializer(runtime,state,pin,*,contract,plant_pin,expected_iteration=None):
    active_backend(contract)
    if state['metadata'].get('collision_backend')!=contract or state['native']['metadata'].get('collision_backend')!=contract:
        raise ValueError('Actor source belongs to another backend')
    if expected_iteration is not None and (state['iteration'],state['optimizer_steps'],state['training_transitions'])!=(expected_iteration,expected_iteration*32,expected_iteration*2048):
        raise ValueError('Actor counter budget differs')
    actor,audit=actor_for_fresh_live_W1(runtime,state,pin,plant_pin)
    audit.update(collision_backend_id=contract['backend_id'],runtime_class=type(runtime).__module__+'.'+type(runtime).__qualname__,
        backend_binding_verified=True,old_native_backend_imported=False)
    return actor,audit


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());budget=study_budget(cfg)
    if cfg['evaluation_features']!='LIVE':raise ValueError('LIVE final execution required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['objective_config','backend_manifest','qualification_ledger','qualification_report','protocol','evaluation_protocol','gates','plant_context_report','registration','preparation_report','training_report']}
    contract=load_contract(paths['backend_manifest']);backend_id=contract['backend_id']
    qp=json.loads(paths['qualification_report'].read_text());ql=json.loads(paths['qualification_ledger'].read_text());qs=json.loads(verify_file_pin(ql['summary']).read_text())
    if (qp['passed'],qp['total'],qp['actual_native_integrations'],qp['optimizer_steps'])!=(6,6,42240,128) or qs['passed']!=qs['total'] or qp['backend_manifest']!=file_pin(paths['backend_manifest']):
        raise ValueError('Completed same-backend full integration evidence required')
    verify_source_snapshot(json.loads(verify_file_pin(qp['source_receipt']).read_text()),live=True)
    input_keys=['objective_config','backend_manifest','qualification_ledger','qualification_report','protocol','evaluation_protocol','gates','plant_context_report']
    inputs={k:file_pin(paths[k]) for k in input_keys};contexts=context_pins(json.loads(paths['plant_context_report'].read_text()))
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp]+[paths[k] for k in input_keys if k not in ['qualification_ledger','qualification_report','plant_context_report']]
    sources += [ROOT/'tools'/n for n in ['preview_pen_support_pilot.py','render_mug_progress_video.py','evaluate_pen_long_control.py','train_native_pen_ppo.py']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.mode=='register':
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh whole-study registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='fresh_backend_control_study_registration:1',config=file_pin(cp),inputs=inputs,budget=budget,backend_id=backend_id,
            contexts=contexts,future_preparation_report=str(paths['preparation_report']),future_training_report=str(paths['training_report']),source_receipt=file_pin(out/'source_receipt.json'),
            final_selection='Final2000 only,INITIAL/R/S x3,repeat0 actual saved-pose video. No early checkpoint/policy selection.'))
        print(json.dumps(dict(registration=file_pin(paths['registration']),budget=budget)),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['budget']!=budget or reg['backend_id']!=backend_id or reg['contexts']!=contexts:raise ValueError('Registered study inputs changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh stage output required')
    if a.mode=='prepare' and out!=paths['preparation_report'].parent:raise ValueError('Registered preparation path required')
    if a.mode=='train' and out!=paths['training_report'].parent:raise ValueError('Registered training path required')
    out.mkdir();receipt=freeze(out);counter=WorkCounter();runtime=learner=None;phase='construction';folder=out;monitor=None
    from mjlab.sim.sim import Simulation
    original=Simulation.step
    def step(sim):return counter.native_call(original,sim)
    Simulation.step=step
    try:
        torch.set_num_threads(4);_,ppo=pilot_settings(paths['objective_config'])
        with backend_scope(paths['backend_manifest']) as backend,torch.jit.optimized_execution(False):
            try:
                if a.mode=='prepare':
                    runtime=BackendSupportPenRuntime(paths['objective_config'],32);learner=BackendSupportPenPPO(runtime,ppo,'RESIDUAL');phase='fresh_calibration'
                    learner.calibrate(8);common,state=save(learner,receipt,out/'common_initial.pt');require_fresh_common(state,contract)
                    saved=torch.load(verify_file_pin(common),map_location='cpu',weights_only=False)
                    checks=dict(calibration_native=counter.native==2048,no_optimizer=counter.optimizer==0,full_CPU_readback=saved_bytes_equal(state,saved),common_untrained=not state['optimizer']['state'],no_training_transitions=state['training_transitions']==0)
                    result=dict(schema='fresh_backend_common_preparation:1',config=file_pin(cp),registration=file_pin(paths['registration']),backend_id=backend_id,common=common,
                        checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),scope='New backend calibrated common source,one preparation cost;no old native import or policy-performance evaluation.')
                elif a.mode=='qualify-w1':
                    preparation=json.loads(paths['preparation_report'].read_text());pin=file_pin(a.checkpoint.resolve()) if a.checkpoint else preparation['common']
                    state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
                    if state['iteration'] not in [0,1]:raise ValueError('Unstepped admission of common/first checkpoint only')
                    runtime=BackendSupportPenRuntime(paths['objective_config'],1);phase='fresh_LIVE_W1_actor_admission'
                    actor,audit=actor_initializer(runtime,state,pin,contract=contract,plant_pin=contexts['RESIDUAL'],expected_iteration=state['iteration'])
                    checks=dict(no_native=counter.native==0,no_optimizer=counter.optimizer==0,physical_model=audit['full_W1_model_bytes_equal'],actor_norm=audit['actor_normalizer_bytes_equal'],same_backend=audit['backend_binding_verified'])
                    result=dict(schema='fresh_backend_W1_actor_admission:1',config=file_pin(cp),registration=file_pin(paths['registration']),backend_id=backend_id,audit=audit,
                        checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),policy_performance_evaluated=False)
                elif a.mode=='train':
                    prep=json.loads(paths['preparation_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prep['source_receipt']).read_text()),live=True)
                    if prep['backend_id']!=backend_id or prep['registration']!=file_pin(paths['registration']) or (prep['actual_native_integrations'],prep['optimizer_steps'],prep['passed'],prep['total'])!=(2048,0,5,5):raise ValueError('Registered fresh common preparation required')
                    common=torch.load(verify_file_pin(prep['common']),map_location='cpu',weights_only=False);require_fresh_common(common,contract)
                    monitor=GPUMonitor();monitor.start();arms=[]
                    for arm_index,arm in enumerate(cfg['arms']):
                        phase=arm+'_fresh_initial_load';folder=out/arm;folder.mkdir();n0,u0=counter.native,counter.optimizer
                        runtime=BackendSupportPenRuntime(paths['objective_config'],32);learner=BackendSupportPenPPO(runtime,ppo,arm)
                        fork,changes=fork_initial_state(common,learner.metadata());audit=learner.load(fork)
                        if not all(v for v in audit.values() if type(v)is bool) or not audit['native']['all_array_bytes_equal'] or not audit['native']['all_scalar_values_equal']:raise ValueError('Fresh calibrated common load differs')
                        counter.bind_optimizer(learner.optimizer);initial,_=save(learner,receipt,folder/'initial_state.pt')
                        write(folder/'initialization.json',dict(common=prep['common'],patches=changes,load=audit,modules_identical=tensor_tree_equal(common['modules'],fork['modules']),observations_identical=tensor_tree_equal(common['observations'],fork['observations'])))
                        times=[];last=initial
                        for iteration in range(1,2001):
                            tick=time.perf_counter();phase=f'{arm}_collect_{iteration}';t=time.perf_counter();learner.collect(64);collect_s=time.perf_counter()-t
                            phase=f'{arm}_health_{iteration}';health=learner.consistency();write(folder/f'collector_health_{iteration:04d}.json',health)
                            if health['passed']!=health['total']:raise ValueError('Collector failed before update: '+repr([k for k,v in health['checks'].items() if not v]))
                            due=iteration==1 or iteration%100==0 or iteration==2000
                            if due:torch.save(cpu_tree(learner.memory),folder/f'memory_{iteration:04d}.pt')
                            phase=f'{arm}_update_{iteration}';before=counter.optimizer;stats=learner.update()
                            if counter.optimizer-before!=32 or stats['optimizer_steps']!=32 or not finite_training_state(learner):raise ValueError('Nonfinite or incorrectly counted Adam update')
                            checkpoint=None;readback=None
                            if due:
                                phase=f'{arm}_checkpoint_{iteration}';checkpoint,captured=save(learner,receipt,folder/f'checkpoint_{iteration:04d}.pt')
                                if iteration==1:
                                    saved=torch.load(verify_file_pin(checkpoint),map_location='cpu',weights_only=False)
                                    readback=dict(full_CPU_bytes=saved_bytes_equal(captured,saved),counters=(saved['iteration'],saved['optimizer_steps'],saved['training_transitions'])==(1,32,2048))
                                    if not all(readback.values()):raise ValueError('First checkpoint serialization differs')
                                last=checkpoint
                            elapsed=time.perf_counter()-tick;times.append(elapsed);typical=float(np.median(times[-50:]));m=learner.memory
                            row=dict(arm=arm,iteration=iteration,checkpoint=checkpoint,collector=file_pin(folder/f'collector_health_{iteration:04d}.json'),full_first_readback=readback,collect_wall_s=collect_s,loop_wall_s=elapsed,
                                actual_native_integrations=counter.native-n0,optimizer_steps=learner.optimizer_steps,training_transitions=learner.training_transitions,
                                terminated=int(m['terminated'].sum()),geometry_failure_fraction=float(m['contact_objective_geometry_failure'].float().mean()),
                                estimated_total_remaining_loop_s=((2000-iteration)+(1-arm_index)*2000)*typical,
                                ETA_scope='Median up to50 complete loops incl due saves;excludes next-arm construction,final execution/render and future changing contact cost.')
                            write(folder/f'health_{iteration:04d}.json',row)
                            print(json.dumps(dict(arm=arm,iteration=iteration,checks=[health['passed'],health['total']],checkpoint=checkpoint,first_readback=readback,estimated_total_remaining_loop_s=row['estimated_total_remaining_loop_s'])),flush=True)
                        if (counter.native-n0,counter.optimizer-u0,learner.training_transitions)!=(32768000,64000,4096000):raise ValueError('Per-arm registered work differs')
                        arms.append(dict(arm=arm,iterations=2000,common=prep['common'],initial=initial,checkpoint=last,actual_native_integrations=counter.native-n0,optimizer_steps=learner.optimizer_steps,training_transitions=learner.training_transitions,
                            initialization=file_pin(folder/'initialization.json'),final_health=file_pin(folder/'health_2000.json')))
                        runtime.close();runtime=learner=None
                    monitor.stop();telemetry=dict(samples=copy.deepcopy(monitor.samples),errors=copy.deepcopy(monitor.errors));monitor=None
                    checks=dict(both_final_arms=len(arms)==2,native_total=counter.native==65536000,Adam_total=counter.optimizer==128000,no_partial_calls=counter.metadata()['completed_call_accounting_exact'],common_cost=prep['actual_native_integrations']==2048)
                    result=dict(schema='fresh_backend_long_control_training:1',config=file_pin(cp),registration=file_pin(paths['registration']),backend_id=backend_id,preparation=file_pin(paths['preparation_report']),
                        preparation_native_integrations=2048,cohort_native_before_evaluation=65538048,arms=arms,checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),telemetry=telemetry,
                        scope='Fresh same-source new-backend controlled single-seed comparison. Training statistics alone are not strict task benefit,WM/ego or acceleration evidence. Prior failed costs remain in linked ledgers;complete historical total cost is not claimed.')
                elif a.mode=='evaluate':
                    training=json.loads(paths['training_report'].read_text());require_final_training(training,file_pin(cp),backend_id,file_pin(paths['registration']))
                    verify_source_snapshot(json.loads(verify_file_pin(training['source_receipt']).read_text()),live=True)
                    prep=json.loads(paths['preparation_report'].read_text());pins=[prep['common']]+[v['checkpoint'] for v in training['arms']];gates=tomllib.loads(paths['gates'].read_text());cases=[]
                    old_factory=preview.SupportPenRuntime;preview.SupportPenRuntime=BackendSupportPenRuntime
                    try:
                        with torch.no_grad():
                            for repetition in range(3):
                                rep=out/f'REP{repetition}';rep.mkdir()
                                for label,pin in zip(cfg['case_order'],pins):
                                    phase=f'REP{repetition}_{label}';plant=contexts['SUPPORT' if label=='SUPPORT' else 'RESIDUAL']
                                    initializer=partial(actor_initializer,contract=contract,plant_pin=plant,expected_iteration=0 if label=='INITIAL' else 2000)
                                    case=preview.execute(pin,paths['objective_config'],gates,rep/label,actor_initializer=initializer);case.update(name=phase,policy=label,repetition=repetition)
                                    with np.load(verify_file_pin(case['trace'])) as z:case['executed_control_audit']=audit_executed_control(dict(z),case)
                                    if case['executed_control_audit']['passed']!=case['executed_control_audit']['total']:raise ValueError('Executed target/drive recording audit failed')
                                    cases.append(case);print(json.dumps(dict(name=phase,summary=case['summary'])),flush=True)
                    finally:preview.SupportPenRuntime=old_factory
                    if counter.native!=sum(c['native_integrations'] for c in cases):raise ValueError('Independent evaluation count differs')
                    common_s=min(c['summary']['metrics']['actual_duration_s'] for c in cases if c['policy']!='INITIAL');prefixes={}
                    for case in cases:
                        if case['policy']=='INITIAL':continue
                        with np.load(verify_file_pin(case['trace'])) as z:
                            n=round(common_s/.0025);trace={k:z[k][:n] for k in z.files if k.startswith(('post_','pre_')) and k!='pre_action_phase'};trace['overflow']=z['overflow'][:n]
                            reference={k:z[k] for k in ['time_s','COM_world_m','quaternion_wxyz']};prefixes[case['name']]=summarize_trace(trace,reference,.0025,False,gates)['metrics']
                    rendering=preview.render(cases[:3],out,title='新碰撞后端2000轮：真实仿真执行')
                    result=dict(schema='fresh_backend_long_control_evaluation:1',config=file_pin(cp),registration=file_pin(paths['registration']),backend_id=backend_id,training=file_pin(paths['training_report']),cases=cases,
                        aggregate={label:dict(durations_s=[c['summary']['metrics']['actual_duration_s'] for c in cases if c['policy']==label],qualified_count=sum(c['summary']['qualified'] for c in cases if c['policy']==label)) for label in cfg['case_order']},
                        trained_common_horizon_s=common_s,common_horizon_metrics=prefixes,rendering=rendering,**counter.metadata(),scope='All9 final fresh LIVE W1 executions;full strict physical task,actual saved-pose video. Controlled viewed single seed/reference,not independent ego/generalization/acceleration evidence.')
                else:raise ValueError('Unknown mode')
                if runtime is not None:runtime.close();runtime=learner=None
                result.update(source_receipt=file_pin(out/'source_receipt.json'),wall_s=time.perf_counter()-start)
                if 'checks' in result and not all(result['checks'].values()):raise ValueError('Stage check failed')
                verify_source_snapshot(receipt,live=True)
                for pin in inputs.values():verify_file_pin(pin)
                write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),mode=a.mode,**counter.metadata())),flush=True)
            except Exception as error:
                base=dict(type=type(error).__name__,message=str(error),phase=phase,**counter.metadata(),source_receipt=file_pin(out/'source_receipt.json'),registration=file_pin(paths['registration']))
                write(out/'failure_exception.json',base)
                try:evidence=persist_failed_learner(learner,folder,receipt)
                except Exception as capture_error:evidence=dict(capture_error=str(capture_error))
                write(out/'failure.json',dict(base,evidence=evidence,backend_scope_active_during_capture=True,wall_s=time.perf_counter()-start));raise
            finally:
                if monitor is not None:monitor.stop()
                for item in list(backend.runtime_set):item.close()
                runtime=learner=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=['register','prepare','train','qualify-w1','evaluate'],required=True)
    p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    if a.mode!='register' and a.output is None:p.error('--output required')
    main(a)
