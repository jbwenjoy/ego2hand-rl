"""Matched source restores and native finite-response comparison,default retained."""
import argparse,copy,json,subprocess,sys,time,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.frozen_pen_actor import frozen_actor_for_current_W1
from supermimic_research.learning.pen_support_ppo import static_contract
from supermimic_research.learning.pen_interventions import intervention_profiles
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,save_physics,restore_source,perturbed_actor_raw
from review_owned_pen_state import compare_states


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['training_report']).resolve();training=json.loads(tp.read_text());oldp=(cp.parent/cfg['W1_context_report']).resolve();old=json.loads(oldp.read_text());op=(cp.parent/cfg['objective_config']).resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['source_phases']!=[10,30,60] or cfg['restore_repeats']!=3:raise ValueError('Fresh registered matched epoch benchmark')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/review_owned_pen_state.py',cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    directions,profiles=intervention_profiles();np.savez_compressed(out/'directions.npz',directions=directions)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;recorder=None;sources=[];states=[];checks={};timings=[];cases=[];initializations=[];phase='source_generation'
    def step(sim):
        nonlocal observed
        if recorder is not None and recorder.enabled:recorder.before_step()
        original(sim);observed+=sim.num_envs
        if recorder is not None and recorder.enabled:recorder.after_step()
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            for arm in training['arms']:
                actor_name=arm['arm'];folder=out/actor_name;folder.mkdir();runtime=SupportPenRuntime(op,1)
                context=next(s['source'] for s in old['sources'] if s['actor']==actor_name and s['phase']==10)
                actor,audit=frozen_actor_for_current_W1(runtime,arm['checkpoint'],context);initializations.append(audit);physics=save_physics(runtime,folder)
                for k in cfg['source_phases']:
                    while int(runtime.motion.time_steps[0])<k:
                        raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean;_,_,term,trunc,_=runtime.step(raw)
                        if bool(term[0] or trunc[0]):raise ValueError('Declared producer source unavailable; preserve failure')
                    c=runtime.contact_views;normal=float(c['Fn6_sum'].sum()/c['counts'][0,0]);geometry=c['geometry_max'][0].cpu().numpy()
                    if not np.all(geometry<=np.array([.002,.0002,.001])) or normal<=.1 or bool(c['errors'].any() or c['global_error'].any()):raise ValueError('Producer source quality failed')
                    raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy();perturbed_actor_raw(raw,np.zeros(20),0.,0)
                    source=dict(schema='owned_trained_actor_probe_source:1',native=native.capture(runtime,receipt),task_static=static_contract(runtime,native.inventory(runtime)),
                        observation={key:v.cpu().clone() for key,v in runtime.observation().items()},base_raw=raw,phase=k,actor=actor_name,actor_checkpoint=arm['checkpoint'])
                    file=folder/f'phase{k:03d}.pt';torch.save(source,file);states.append(source)
                    sources.append(dict(name=f'{actor_name}_P{k:03d}',actor=actor_name,phase=k,source=file_pin(file),model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),physics=physics))
                runtime.close();runtime=None
            prefix_count=observed;checks['prefix_budget']=prefix_count==944;write(out/'sources.json',dict(sources=sources,initializations=initializations,actual_native_integrations=prefix_count))
            runtime=SupportPenRuntime(op,1);recorder=OwnedProbeRecorder(runtime);first=states[0]['native']
            epoch=SourceValidationEpoch(first['sources'],first['engine_sources']);phase='matched_restores'
            with epoch:
                for i,(entry,source) in enumerate(zip(sources,states)):
                    restore_source(runtime,source)
                    for repeat in range(3):
                        for mode in (['STRICT','READ_LOCKED_EPOCH'] if (i+repeat)%2==0 else ['READ_LOCKED_EPOCH','STRICT']):
                            token=None if mode=='STRICT' else epoch;start=time.perf_counter();audit=restore_source(runtime,source,source_epoch=token);wall=time.perf_counter()-start
                            okay=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal'] and all(torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items())
                            checks[f'{entry["name"]}_{mode}_R{repeat}']=okay;timings.append(dict(source=entry['name'],mode=mode,repeat=repeat,wall_s=wall))
                    print(json.dumps(dict(stage='restores',source=entry['name'],last_pair=timings[-2:])),flush=True)
                # Invalid manifests/targets must fail before any native mutation.
                before=native.capture(runtime,receipt);bad=copy.deepcopy(states[-1]);bad['native']['sources']['code_revision']='foreign'
                try:restore_source(runtime,bad,source_epoch=epoch)
                except ValueError:checks['reject_foreign_manifest']=True
                else:checks['reject_foreign_manifest']=False
                bad=copy.deepcopy(states[-1]);path=next(p for p in bad['task_static']['tensors'] if bad['native']['python']['tensors'][p].dtype==torch.float64);bad['native']['python']['tensors'][path].flatten()[0]+=1
                try:restore_source(runtime,bad,source_epoch=epoch)
                except ValueError:checks['reject_target_corruption']=True
                else:checks['reject_target_corruption']=False
                after=native.capture(runtime,receipt);checks.update({'negative_atomic_'+k:v for k,v in compare_states(before,after).items()});checks['restore_only_no_physics']=observed==prefix_count
                phase='native_physical_probes'
                for i,(entry,source) in enumerate(zip(sources,states)):
                    baseline=None;f=runtime.features;m=runtime.env.sim.mj_model;b=f.body;qa=f.qa;da=int(m.jnt_dofadr[m.body_jntadr[b]]);aids=runtime.action._position_ctrl_ids.cpu().numpy()
                    for j,profile in enumerate(profiles):
                        direction=np.zeros(20) if profile['direction']<0 else directions[profile['direction']];raw=perturbed_actor_raw(source['base_raw'],direction,profile['epsilon'],profile['sign'])
                        for mode in (['STRICT','READ_LOCKED_EPOCH'] if (i+j)%2==0 else ['READ_LOCKED_EPOCH','STRICT']):
                            audit=restore_source(runtime,source,source_epoch=None if mode=='STRICT' else epoch)
                            folder=out/entry['actor']/f'PHASE{entry["phase"]:03d}'/mode/profile['name'];folder.mkdir(parents=True)
                            recorder.begin(raw);_,_,term,trunc,_=runtime.step(torch.tensor(raw,device=runtime.device)[None]);packet=recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
                            pd=packet['post_ctrl'][0,aids]
                            if baseline is None:baseline=pd.copy()
                            pd_error=float(abs(pd-baseline-profile['sign']*profile['epsilon']*direction).max())
                            np.savez_compressed(folder/'trace.npz',**packet);np.savez_compressed(folder/'window.npz',**w)
                            case=dict(name=entry['name']+'_'+mode+'_'+profile['name'],source_name=entry['name'],mode=mode,profile=profile,role=cfg['role'],
                                restore_exact=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal'],PD_error_rad=pd_error,
                                native_integrations=len(packet['post_qpos']),model=entry['model'],physical_context=entry['physical_context'],physics=entry['physics'],dt=.0025,
                                model_layout=dict(object_body=b,object_qposadr=qa,object_dofadr=da,COM_BODY=[0.,0,0]),qids=f.qids.cpu().numpy().tolist(),dofs=f.dofs.cpu().numpy().tolist(),aids=aids.tolist(),
                                trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),source=entry['source'],terminated=bool(term[0]),truncated=bool(trunc[0]))
                            cases.append(case)
                    write(out/'progress.json',dict(sources=sources,cases=cases,actual_native_integrations=observed));print(json.dumps(dict(stage='physical',source=entry['name'],cases=len(cases))),flush=True)
            checks['epoch_validated']=epoch.validated;before=native.capture(runtime,receipt)
            try:restore_source(runtime,states[0],source_epoch=epoch)
            except ValueError:checks['reject_expired_epoch']=True
            else:checks['reject_expired_epoch']=False
            after=native.capture(runtime,receipt);checks.update({'expired_atomic_'+k:v for k,v in compare_states(before,after).items()})
            checks.update(declared_sources=len(sources)==6,declared_timings=len(timings)==36,declared_physical_cases=len(cases)==324,actual_total=observed==3536,
                every_source_exact=all(c['restore_exact'] for c in cases),every_PD_mapping=all(c['PD_error_rad']<=2e-7 for c in cases))
            verify_source_snapshot(receipt,live=True)
            total={mode:sum(t['wall_s'] for t in timings if t['mode']==mode) for mode in cfg['modes']};result=dict(schema='source_epoch_matched_benchmark:1',config=file_pin(cp),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),
                training=file_pin(tp),historical_W1_context_report=file_pin(oldp),initializations=initializations,sources=sources,timings=timings,restore_only_total_s=total,
                restore_only_ratio=total['STRICT']/total['READ_LOCKED_EPOCH'],epoch=epoch.audit(),cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),
                producer_native_integrations=prefix_count,actual_native_integrations=observed,optimizer_steps=0,role=cfg['role'],scope=cfg['scope'])
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,timing=total,ratio=result['restore_only_ratio'],epoch=result['epoch'])),flush=True)
            if not all(checks.values()):raise ValueError('Retain benchmark contract discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
