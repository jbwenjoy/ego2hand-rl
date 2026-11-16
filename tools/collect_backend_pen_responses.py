"""Registered final-policy probes on the explicit collision backend; no fitting."""
import argparse
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
from supermimic_research.learning.backend_response_study import response_spec,require_final_response_actor,require_evaluated_cohort
from supermimic_research.learning.pen_interventions import prefix_noise_schedule
from supermimic_research.learning.pen_w1_evaluation import actor_for_fresh_live_W1
from supermimic_research.learning.pen_support_ppo import static_contract
from supermimic_research.learning.pen_matched_recovery import WorkCounter,write_metadata as write
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.backend_pen_probe import SCHEMA,require_backend_probe,restore_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,save_physics,source_frame_gap,perturbed_actor_raw
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from evaluate_pen_long_control import context_pins


def read(p):return json.loads(Path(p).read_text())
def paths(cp,cfg):
    return {k:(cp.parent/cfg[k]).resolve() for k in ['backend_manifest','control_study_config','training_report','evaluation_report',
        'control_registration','prior_ledger','objective_config','frozen_W1_context_report','protocol','registration','paired_config','output']}


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());directions,profiles,budget=response_spec(cfg);p=paths(cp,cfg)
    paired=tomllib.loads(p['paired_config'].read_text());_,_,paired_budget=response_spec(paired);other=paths(p['paired_config'],paired)
    common_keys=['backend_manifest','control_study_config','training_report','evaluation_report','control_registration','prior_ledger','objective_config','frozen_W1_context_report','protocol','registration']
    if cfg['role']==paired['role'] or any(p[k]!=other[k] for k in common_keys) or other['paired_config']!=cp:raise ValueError('Exact mutually linked TRAIN/NEVER_FIT pair required')
    contract=load_contract(p['backend_manifest']);control_reg=read(p['control_registration'])
    if control_reg['backend_id']!=contract['backend_id'] or control_reg['config']!=file_pin(p['control_study_config']):raise ValueError('Same registered control study/backend required')
    verify_source_snapshot(read(verify_file_pin(control_reg['source_receipt'])),live=True)
    inputs={k:file_pin(p[k]) for k in common_keys if k not in ['training_report','evaluation_report','registration']}
    specs={v['role']:file_pin(path) for path,v in [(cp,cfg),(p['paired_config'],paired)]}
    budgets={cfg['role']:budget,paired['role']:paired_budget};contexts=context_pins(read(p['frozen_W1_context_report']))
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/evaluate_pen_long_control.py',cp,p['paired_config']]
    sources += [p[k] for k in ['backend_manifest','control_study_config','objective_config','protocol']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(out/'source_receipt.json',receipt);return receipt
    if a.mode=='register':
        out=p['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh paired-response registration required')
        out.mkdir();freeze(out)
        write(p['registration'],dict(schema='backend_physical_response_registration:1',inputs=inputs,specs=specs,budgets=budgets,contexts=contexts,backend_id=contract['backend_id'],
            future_training_report=str(p['training_report']),future_evaluation_report=str(p['evaluation_report']),outputs={cfg['role']:str(p['output']),paired['role']:str(other['output'])},
            source_receipt=file_pin(out/'source_receipt.json'),scope='Design only,before complete final trial/evaluation. No native/NN/Actor work or response admission.'))
        print(json.dumps(dict(registration=file_pin(p['registration']),budgets=budgets)),flush=True);return
    reg=read(p['registration'])
    if reg['inputs']!=inputs or reg['specs']!=specs or reg['budgets']!=budgets or reg['contexts']!=contexts or reg['backend_id']!=contract['backend_id']:raise ValueError('Registered response design changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    training=read(p['training_report']);evaluation=read(p['evaluation_report'])
    require_evaluated_cohort(training,evaluation,file_pin(p['control_study_config']),contract['backend_id'],file_pin(p['control_registration']),file_pin(p['training_report']))
    for r in [training,evaluation]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    out=p['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registered cohort output required')
    out.mkdir();receipt=freeze(out);np.savez_compressed(out/'directions.npz',directions=directions)
    schedules={s:prefix_noise_schedule(s,max(cfg['source_phases'])-1,cfg['prefix_raw_sigma']) for s in cfg['noise_seeds']}
    np.savez_compressed(out/'prefix_noise.npz',**{f'seed{s}':v for s,v in schedules.items()})
    from mjlab.sim.sim import Simulation
    original=Simulation.step;counter=WorkCounter();recorder=runtime=None;phase='construction';sources_saved=[];cases=[];producers=[]
    def step(sim):
        if recorder is not None and recorder.enabled:recorder.before_step()
        result=counter.native_call(original,sim)
        if recorder is not None and recorder.enabled:recorder.after_step()
        return result
    Simulation.step=step
    def make_runtime(arm):
        rt=BackendSupportPenRuntime(p['objective_config'],1);rt.set_fixed_frame_features(False);rt.support_weight=float(rt.objective_cfg['support_weights'][arm]);return rt
    try:
        torch.set_num_threads(4)
        with backend_scope(p['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for arm in training['arms']:
                    label=arm['arm'];saved=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False);require_final_response_actor(saved,contract,label)
                    for seed in cfg['noise_seeds']:
                        episode=f'{label}_N{seed}';folder=out/episode;folder.mkdir();phase=episode+'_producer'
                        runtime=make_runtime(label);actor,actor_audit=actor_for_fresh_live_W1(runtime,saved,arm['checkpoint'],contexts[label])
                        backend.verify(cold=True);actor_audit.update(collision_backend_id=contract['backend_id'],backend_binding_verified=True,old_native_backend_imported=False)
                        physics=save_physics(runtime,folder);recorder=OwnedProbeRecorder(runtime);terminated=False;before=counter.native;prefix_actions=[]
                        for k in cfg['source_phases']:
                            while int(runtime.motion.time_steps[0])<k and not terminated:
                                raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean;control=int(runtime.motion.time_steps[0])-1
                                mean=raw[0].cpu().numpy().copy();raw=raw+torch.tensor(schedules[seed][control],device=runtime.device)[None]
                                prefix_actions.append(dict(control_index=control,phase=int(runtime.motion.time_steps[0]),mean_raw=mean.tolist(),submitted_raw=raw[0].cpu().numpy().tolist(),clamped_components=int((raw.abs()>1).sum())))
                                _,_,term,trunc,_=runtime.step(raw);terminated=bool(term[0] or trunc[0])
                            sf=folder/f'PHASE{k:03d}';sf.mkdir();name=f'{episode}_P{k:03d}'
                            if terminated:
                                sources_saved.append(dict(name=name,actor=label,episode=episode,noise_seed=seed,phase=k,available=False,qualified=False,reason='Producer terminated before source;no replacement'));continue
                            d=runtime.env.sim.wp_data;f=runtime.features;c={key:v.cpu().numpy().copy() for key,v in runtime.contact_views.items()}
                            raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy();gap=source_frame_gap(d.qpos.numpy()[0],f.qa,d.xmat.numpy()[0,f.body],d.xipos.numpy()[0,f.body])
                            geometry=c['geometry_max'][0];count=int(c['counts'][0,0]);normal=float(c['Fn6_sum'].sum()/count) if count else 0.
                            quality=dict(nonterminal=not bool(runtime.pending.any()),eight_previous_samples=count==8,geometry=bool(np.all(geometry<=np.array([cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))),
                                nonhand=float(c['nonhand_max_sum'][0,0])<=cfg['nonhand_N'],contact_load=normal>cfg['source_mean_hand_load_N'],no_metric_error=not bool(c['errors'].any() or c['global_error'].any()),
                                finite=bool(np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all()),no_aid=not np.any(d.qfrc_applied.numpy()) and not np.any(d.xfrc_applied.numpy()),unclipped_base_raw=bool(np.max(abs(raw))<=1))
                            state=dict(schema=SCHEMA,role=cfg['role'],collision_backend=contract,actor_budget=(2000,64000,4096000),native=native.capture(runtime,receipt),task_static=static_contract(runtime,native.inventory(runtime)),
                                observation={key:v.cpu().clone() for key,v in runtime.observation().items()},base_raw=raw,phase=k,actor=label,actor_checkpoint=arm['checkpoint'],
                                source_qpos_R=gap['qpos_R'],source_qpos_COM=gap['qpos_COM'],source_cached_R=d.xmat.numpy()[0,f.body].copy(),source_cached_COM=d.xipos.numpy()[0,f.body].copy(),episode=episode,noise_seed=seed)
                            require_backend_probe(state,contract);torch.save(state,sf/'source.pt')
                            entry=dict(name=name,actor=label,episode=episode,noise_seed=seed,phase=k,source_time_s=(k-1)*.02,available=True,qualified=all(quality.values()),quality=quality,previous_geometry=geometry.tolist(),
                                previous_mean_hand_normal_N=normal,cached_rotation_gap_rad=gap['cached_rotation_gap_rad'],cached_COM_gap_m=gap['cached_COM_gap_m'],source=file_pin(sf/'source.pt'),model=file_pin(folder/'model.mjb'),
                                physical_context=file_pin(folder/'physical_context.npz'),physics=physics,actor_checkpoint=arm['checkpoint'],collision_backend_id=contract['backend_id'])
                            sources_saved.append(entry);write(sf/'source.json',entry)
                        producer_path=folder/'producer.json';write(producer_path,dict(native_integrations=counter.native-before,no_post_start_resets=True,final_phase=int(runtime.motion.time_steps[0]),terminated=terminated,
                            actor_audit=actor_audit,episode=episode,noise_seed=seed,prefix_actions=prefix_actions));producers.append(file_pin(producer_path));runtime.close();runtime=recorder=None
                producer_native=counter.native;write(out/'source_bank.json',dict(sources=sources_saved,actual_native_integrations=producer_native,role=cfg['role'],backend_id=contract['backend_id']))
                for label in cfg['actors']:
                    phase=label+'_counterfactual';runtime=make_runtime(label);recorder=OwnedProbeRecorder(runtime)
                    for entry in [s for s in sources_saved if s['actor']==label]:
                        baseline=None;source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False) if entry['available'] else None
                        f=runtime.features;m=runtime.env.sim.mj_model;b=f.body;qa=f.qa;da=int(m.jnt_dofadr[m.body_jntadr[b]]);aids=runtime.action._position_ctrl_ids.cpu().numpy()
                        for profile in profiles:
                            phase=entry['name']+'_'+profile['name'];case=dict(name=phase,source_name=entry['name'],actor=label,episode=entry['episode'],noise_seed=entry['noise_seed'],phase=entry['phase'],profile=profile,role=cfg['role'],collision_backend_id=contract['backend_id'])
                            if not entry['qualified']:
                                case.update(available=False,qualified=False,reason='Source quality exclusion;no replacement',native_integrations=0);cases.append(case);continue
                            direction=np.zeros(20) if profile['direction']<0 else directions[profile['direction']]
                            try:raw=perturbed_actor_raw(source['base_raw'],direction,profile['epsilon'],profile['sign'])
                            except ValueError as error:
                                case.update(available=False,qualified=False,reason=str(error),native_integrations=0);cases.append(case);continue
                            audit=restore_backend_probe(runtime,source)
                            if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or any(not torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items()):raise ValueError('Strict source/observation restoration differs')
                            folder=Path(entry['source']['path']).parent/profile['name'];folder.mkdir();recorder.begin(raw);before=counter.native
                            _,_,term,trunc,info=runtime.step(torch.tensor(raw,device=runtime.device)[None]);packet=recorder.finish();window={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
                            actual=packet['post_ctrl'][0,aids]
                            if baseline is None:baseline=actual.copy()
                            error=float(abs(actual-(baseline+profile['sign']*profile['epsilon']*direction)).max());held=bool(np.array_equal(packet['post_ctrl'][:,aids],np.tile(actual,(8,1))))
                            quality=dict(native_count=counter.native-before==8,held_PD=held,unclipped_PD=error<=2e-7,eight_samples=int(window['counts'][0])==8,
                                geometry=bool(np.all(window['geometry_max']<=np.array([cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))),nonhand=float(window['nonhand_max_sum'][0])<=cfg['nonhand_N'],
                                no_metric_error=not bool(window['errors'].any() or window['global_error'].any()),finite=all(np.isfinite(v).all() for k,v in packet.items() if not k.startswith('legacy_host_')),
                                no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']))
                            np.savez_compressed(folder/'trace.npz',**packet);np.savez_compressed(folder/'window.npz',**window)
                            case.update(available=True,qualified=all(quality.values()),quality=quality,native_integrations=counter.native-before,restore_audit=audit,PD_error_rad=error,actual_PD20=actual.tolist(),base_PD20=baseline.tolist(),
                                trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),source=entry['source'],model=entry['model'],physical_context=entry['physical_context'],physics=entry['physics'],dt=.0025,
                                model_layout=dict(object_body=b,object_qposadr=qa,object_dofadr=da,COM_BODY=[0.,0,0]),qids=f.qids.cpu().numpy().tolist(),dofs=f.dofs.cpu().numpy().tolist(),aids=aids.tolist(),
                                terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()})
                            cases.append(case);write(folder/'report.json',case)
                        write(out/'progress.json',dict(sources=sources_saved,cases=cases,**counter.metadata()))
                        print(json.dumps(dict(source=entry['name'],qualified=entry['qualified'],available_branches=sum(c['source_name']==entry['name'] and c['available'] for c in cases),**counter.metadata())),flush=True)
                    runtime.close();runtime=recorder=None
                checks=dict(declared_sources=len(sources_saved)==cfg['expected_sources'],declared_cases=len(cases)==cfg['expected_cases'],all_actual_steps_accounted=counter.native==producer_native+sum(c['native_integrations'] for c in cases),
                    all_available_restore_exact=all(c['restore_audit']['all_array_bytes_equal'] and c['restore_audit']['all_scalar_values_equal'] for c in cases if c['available']),all_available_native_count=all(c['native_integrations']==8 for c in cases if c['available']),
                    bounded_work=counter.native<=budget['max_total_native'],no_optimizer=counter.optimizer==0,exact_calls=counter.metadata()['completed_call_accounting_exact'])
                verify_source_snapshot(receipt,live=True);backend.verify(cold=True);write(out/'backend_audit.json',backend.audit())
                result=dict(schema='backend_controlled_response_probe:1',config=file_pin(cp),registration=file_pin(p['registration']),training=file_pin(p['training_report']),evaluation=file_pin(p['evaluation_report']),
                    backend_manifest=file_pin(p['backend_manifest']),collision_backend=contract,protocol=file_pin(p['protocol']),source_receipt=file_pin(out/'source_receipt.json'),directions=file_pin(out/'directions.npz'),
                    prefix_noise=file_pin(out/'prefix_noise.npz'),source_bank=file_pin(out/'source_bank.json'),producers=producers,sources=sources_saved,cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),
                    qualified_sources=sum(s['qualified'] for s in sources_saved),available_cases=sum(c['available'] for c in cases),online_geometry_qualified_cases=sum(c['qualified'] for c in cases),
                    producer_native_integrations=producer_native,**counter.metadata(),budget=budget,role=cfg['role'],restore_mode='STRICT',fixed_frame_features='LIVE',backend_audit=file_pin(out/'backend_audit.json'),
                    scales_fitted=False,CHECK_response_rows_used=0,validation_rows_used_for_TRAIN=0,
                    wall_s=time.perf_counter()-start,scope=cfg['scope'])
                if not all(checks.values()):raise ValueError('Response cohort accounting differs: '+repr(checks))
                write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],**counter.metadata())),flush=True)
            except Exception as error:
                failure=dict(type=type(error).__name__,message=str(error),phase=phase,**counter.metadata(),backend_scope_active_during_capture=True,resumable=False)
                write(out/'failure_exception.json',failure)
                if recorder is not None and recorder.enabled:
                    try:np.savez_compressed(out/'failure_partial_packet_NOT_RESUMABLE.npz',**recorder.finish());failure['partial_packet']=file_pin(out/'failure_partial_packet_NOT_RESUMABLE.npz')
                    except Exception as capture_error:failure['packet_capture_error']=str(capture_error)
                if runtime is not None:
                    try:
                        raw={}
                        for name,root in [('model',runtime.env.sim.wp_model),('data',runtime.env.sim.wp_data)]:
                            arrays,scalars=model_leaves(root);raw[name]=dict(arrays={k:v.numpy().copy() for k,v in arrays.items()},scalars=scalars)
                        torch.save(dict(resumable=False,native_raw=raw,backend=contract),out/'failure_native_NOT_RESUMABLE.pt');failure['native']=file_pin(out/'failure_native_NOT_RESUMABLE.pt')
                    except Exception as capture_error:failure['native_capture_error']=str(capture_error)
                write(out/'failure.json',dict(failure,sources=sources_saved,cases=cases,wall_s=time.perf_counter()-start));raise
            finally:
                for item in list(backend.runtime_set):item.close()
                runtime=recorder=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=['register','collect'],required=True);main(p.parse_args())
