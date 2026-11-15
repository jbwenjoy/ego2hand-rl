"""Uninterrupted trained-actor source bank, then full-native same-source probes."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_interventions import intervention_profiles
from supermimic_research.learning.pen_support_ppo import static_contract,CENTERED_SCHEMA
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,save_physics,source_frame_gap,restore_source,perturbed_actor_raw


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['training_report']).resolve();training=json.loads(tp.read_text());op=(cp.parent/cfg['objective_config']).resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['source_phases']!=[10,20,30,40,60,75] or cfg['actors']!=['RESIDUAL','SUPPORT'] or cfg['PD_eps_rad']!=[.002,.005] or cfg['repeats']!=3:raise ValueError('Fresh registered324-branch study')
    if training['config']!=file_pin(op) or training['passed']!=training['total']:raise ValueError('Completed registered training required')
    verify_source_snapshot(json.loads(verify_file_pin(training['source_receipt']).read_text()),live=True)
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    directions,profiles=intervention_profiles();np.savez_compressed(out/'directions.npz',directions=directions)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;recorder=None;runtime=None;sources=[];cases=[];phase='construction'
    def step(sim):
        nonlocal observed
        if recorder is not None and recorder.enabled:recorder.before_step()
        original(sim);observed+=sim.num_envs
        if recorder is not None and recorder.enabled:recorder.after_step()
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            # Finish the entire two-actor bank BEFORE restoring any source.
            for arm in training['arms']:
                label=arm['arm'];folder=out/label;folder.mkdir();phase=label+'_producer';saved=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False)
                if saved['schema']!=CENTERED_SCHEMA:raise ValueError('Registered training4 actor required')
                runtime=SupportPenRuntime(op,1);runtime.support_weight=float(saved['metadata']['support_weight']);settings=saved['config']
                actor=GaussianPPOActor(161,20,init_std=settings['ppo_init_std'],std_range=(settings['ppo_std_min'],settings['ppo_std_max'])).to(runtime.device);actor.load_state_dict(saved['modules']['actor']);actor.eval()
                physics=save_physics(runtime,folder);recorder=OwnedProbeRecorder(runtime);terminated=False;before=observed
                current_arrays=runtime.env.sim.wp_model
                for key in ['body_mass','body_inertia','body_ipos','body_iquat','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','jnt_range','dof_damping','dof_armature']:
                    current=getattr(current_arrays,key).numpy();old=saved['native']['native']['model']['arrays'][key]
                    if not np.array_equal(np.broadcast_to(current,old.shape),old):raise ValueError('Source producer physics differs '+key)
                for k in cfg['source_phases']:
                    while int(runtime.motion.time_steps[0])<k and not terminated:
                        raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean
                        _,_,term,trunc,_=runtime.step(raw);terminated=bool(term[0] or trunc[0])
                    source_folder=folder/f'PHASE{k:03d}';source_folder.mkdir();name=f'{label}_P{k:03d}'
                    if terminated:
                        sources.append(dict(name=name,actor=label,phase=k,available=False,qualified=False,reason='Producer terminated before source; no replacement'))
                        continue
                    d=runtime.env.sim.wp_data;f=runtime.features;c={key:v.cpu().numpy().copy() for key,v in runtime.contact_views.items()}
                    raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy()
                    gap=source_frame_gap(d.qpos.numpy()[0],f.qa,d.xmat.numpy()[0,f.body],d.xipos.numpy()[0,f.body])
                    geometry=c['geometry_max'][0];normal=float(c['Fn6_sum'].sum()/c['counts'][0,0])
                    quality=dict(nonterminal=not bool(runtime.pending.any()),eight_previous_samples=int(c['counts'][0,0])==8,
                        geometry=bool(np.all(geometry<=np.array([cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))),
                        nonhand=float(c['nonhand_max_sum'][0,0])<=cfg['nonhand_N'],contact_load=normal>cfg['source_mean_hand_load_N'],
                        no_metric_error=not bool(c['errors'].any() or c['global_error'].any()),finite=bool(np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all()),
                        no_aid=not np.any(d.qfrc_applied.numpy()) and not np.any(d.xfrc_applied.numpy()),unclipped_base_raw=bool(np.max(abs(raw))<=1))
                    state=dict(schema='owned_trained_actor_probe_source:1',native=native.capture(runtime,receipt),task_static=static_contract(runtime,native.inventory(runtime)),
                        observation={key:value.cpu().clone() for key,value in runtime.observation().items()},base_raw=raw,phase=k,actor=label,actor_checkpoint=arm['checkpoint'],
                        source_qpos_R=gap['qpos_R'],source_qpos_COM=gap['qpos_COM'],source_cached_R=d.xmat.numpy()[0,f.body].copy(),source_cached_COM=d.xipos.numpy()[0,f.body].copy())
                    torch.save(state,source_folder/'source.pt')
                    entry=dict(name=name,actor=label,phase=k,source_time_s=(k-1)*.02,available=True,qualified=all(quality.values()),quality=quality,
                        previous_geometry=geometry.tolist(),previous_mean_hand_normal_N=normal,cached_rotation_gap_rad=gap['cached_rotation_gap_rad'],cached_COM_gap_m=gap['cached_COM_gap_m'],
                        source=file_pin(source_folder/'source.pt'),model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),physics=physics,actor_checkpoint=arm['checkpoint'])
                    sources.append(entry);write(source_folder/'source.json',entry)
                prefix=observed-before;write(folder/'producer.json',dict(native_integrations=prefix,no_post_start_resets=True,final_phase=int(runtime.motion.time_steps[0]),terminated=terminated))
                runtime.close();runtime=None;recorder=None
            producer_integrations=observed;write(out/'source_bank.json',dict(sources=sources,actual_native_integrations=producer_integrations,role=cfg['role']))
            for label in cfg['actors']:
                phase=label+'_counterfactual';runtime=SupportPenRuntime(op,1);runtime.support_weight=float(runtime.objective_cfg['support_weights'][label]);recorder=OwnedProbeRecorder(runtime)
                for entry in [s for s in sources if s['actor']==label]:
                    baseline_PD=None;source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False) if entry['available'] else None
                    f=runtime.features;m=runtime.env.sim.mj_model;b=f.body;qa=f.qa;da=int(m.jnt_dofadr[m.body_jntadr[b]]);aids=runtime.action._position_ctrl_ids.cpu().numpy()
                    for profile in profiles:
                        name=entry['name']+'_'+profile['name'];case=dict(name=name,source_name=entry['name'],actor=label,phase=entry['phase'],profile=profile,role=cfg['role'])
                        if not entry['qualified']:
                            case.update(available=False,qualified=False,reason='Source quality exclusion; no replacement',native_integrations=0);cases.append(case);continue
                        direction=np.zeros(20) if profile['direction']<0 else directions[profile['direction']]
                        try:raw=perturbed_actor_raw(source['base_raw'],direction,profile['epsilon'],profile['sign'])
                        except ValueError as error:
                            case.update(available=False,qualified=False,reason=str(error),native_integrations=0);cases.append(case);continue
                        audit=restore_source(runtime,source)
                        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or any(not torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items()):raise ValueError('Source/observation restoration differs')
                        folder=out/label/f'PHASE{entry["phase"]:03d}'/profile['name'];folder.mkdir();recorder.begin(raw);start=observed
                        _,_,term,trunc,info=runtime.step(torch.tensor(raw,device=runtime.device)[None]);packet=recorder.finish();window={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
                        actual=packet['post_ctrl'][0,aids]
                        if baseline_PD is None:baseline_PD=actual.copy()
                        intended=baseline_PD+profile['sign']*profile['epsilon']*direction;error=float(abs(actual-intended).max())
                        held=bool(np.array_equal(packet['post_ctrl'][:,aids],np.tile(actual,(8,1))))
                        quality=dict(native_count=observed-start==8,held_PD=held,unclipped_PD=error<=2e-7,eight_samples=int(window['counts'][0])==8,
                            geometry=bool(np.all(window['geometry_max']<=np.array([cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))),
                            nonhand=float(window['nonhand_max_sum'][0])<=cfg['nonhand_N'],no_metric_error=not bool(window['errors'].any() or window['global_error'].any()),
                            finite=all(np.isfinite(v).all() for k,v in packet.items() if not k.startswith('legacy_host_')),no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']))
                        np.savez_compressed(folder/'trace.npz',**packet);np.savez_compressed(folder/'window.npz',**window)
                        case.update(available=True,qualified=all(quality.values()),quality=quality,native_integrations=observed-start,restore_audit=audit,PD_error_rad=error,
                            actual_PD20=actual.tolist(),base_PD20=baseline_PD.tolist(),trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),
                            source=entry['source'],model=entry['model'],physical_context=entry['physical_context'],physics=entry['physics'],dt=.0025,
                            model_layout=dict(object_body=b,object_qposadr=qa,object_dofadr=da,COM_BODY=[0.,0,0]),qids=f.qids.cpu().numpy().tolist(),dofs=f.dofs.cpu().numpy().tolist(),aids=aids.tolist(),
                            terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()})
                        cases.append(case);write(folder/'report.json',case)
                    write(out/'progress.json',dict(sources=sources,cases=cases,actual_native_integrations=observed))
                    print(json.dumps(dict(source=entry['name'],source_qualified=entry['qualified'],branches=sum(c['source_name']==entry['name'] and c['available'] for c in cases),observed_native=observed)),flush=True)
                runtime.close();runtime=None;recorder=None
            checks=dict(declared_sources=len(sources)==12,declared_cases=len(cases)==324,all_actual_steps_accounted=observed==producer_integrations+sum(c['native_integrations'] for c in cases),
                all_available_restore_exact=all(c['restore_audit']['all_array_bytes_equal'] and c['restore_audit']['all_scalar_values_equal'] for c in cases if c['available']),
                all_available_native_count=all(c['native_integrations']==8 for c in cases if c['available']))
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='trained_pen_same_source_response_probe:1',config=file_pin(cp),training=file_pin(tp),source_receipt=file_pin(out/'source_receipt.json'),protocol=file_pin(protocol),directions=file_pin(out/'directions.npz'),
                source_bank=file_pin(out/'source_bank.json'),sources=sources,cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),qualified_sources=sum(s['qualified'] for s in sources),
                available_cases=sum(c['available'] for c in cases),online_geometry_qualified_cases=sum(c['qualified'] for c in cases),producer_native_integrations=producer_integrations,
                actual_native_integrations=observed,optimizer_steps=0,role=cfg['role'],scope=cfg['scope'])
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,qualified_sources=result['qualified_sources'],available_cases=result['available_cases'],geometry_qualified=result['online_geometry_qualified_cases'])),flush=True)
            if not all(checks.values()):raise ValueError('Retain probe contract discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
