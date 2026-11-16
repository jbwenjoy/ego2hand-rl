"""Execute the full declared54-case offline native-screened controller matrix."""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT)]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
from supermimic_research.runtime.pen_live_packets import LivePenPackets
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.learning.pen_live_decision import from_owned_snapshot
from supermimic_research.learning.pen_native_screening import proposal,choose,random_direction
from supermimic_research.learning.pen_path_fit import fixed_spec,data_view,registered_context,restore_arm
from supermimic_research.learning.pen_path_decomposition import gradients
from supermimic_research.learning.backend_reward_gradient import batch_context
from supermimic_research.learning.gpu_monitor import GPUMonitor
from supermimic_research.strict_json import write_json
from run_pen_backend_control_study import actor_initializer
from evaluate_pen_long_control import context_pins


def read(p):return json.loads(Path(p).read_text())


def declarations(design):
    controls=[dict(controller=n,model=None,score_mode=None) for n in design['controls']]
    controls += [dict(controller='WM',model=m,score_mode=s) for m,s in itertools.product(design['models'],design['score_modes'])]
    return [dict(repetition=r,teacher=a,teacher_index=ai,**c,name=f'R{r}_{a}_'+(c['controller'] if c['model'] is None else c['model']+'_'+c['score_mode'])) for r in range(3) for ai,a in enumerate(design['teacher_arms']) for c in controls]


def outcome(case):return dict(physical_valid=bool(case['physical_audit']['valid']),terminated=case['terminated'],truncated=case['truncated'],reward=case['source_reward'])


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text());keys=['design','study_config','model_fit','model_ledger','lifecycle_review','qualification_ledger','protocol','registration','output'];paths={k:(cp.parent/cfg[k]).resolve() for k in keys}
    if (cfg['schema'],cfg['mode'],cfg['reserve_GiB'],cfg['verifier'])!=(1,'FULL54_NATIVE_SCREENED_OFFLINE_EVALUATION',32,'SINGLE_RESOLVE_FULL_REHASH'):raise ValueError('Fixed full54 scope required')
    design=tomllib.loads(paths['design'].read_text());expected=dict(teacher_arms=['RESIDUAL','SUPPORT'],models=['RESPONSE_ONLY','PATH_POINT','PATH_POINT_RESPONSE'],score_modes=['OBSERVABLE','FULL'],controls=['UNMODIFIED_ACTOR','NATIVE_SHAM','NATIVE_RANDOM'],numerical_repetitions=3,reference_controls=368,reference_duration_s=7.36,native_dt_s=.0025,ticks_per_control=8,sampled_RAW_radius=.008,acceptance_reward_margin=.0001,random_direction_seed=9901,cases_per_repetition=18,native_budget_all_repetitions_max=300288)
    if any(design[k]!=v or type(design[k]) is not type(v) for k,v in expected.items()):raise ValueError('Original complete design changed')
    cases=declarations(design);study=tomllib.loads(paths['study_config'].read_text());sp={k:(paths['study_config'].parent/study[k]).resolve() for k in ['backend_manifest','objective_config','training_report','plant_context_report']};contract=load_contract(sp['backend_manifest']);training=read(sp['training_report']);review=read(paths['lifecycle_review']);ledger=read(paths['qualification_ledger']);fit=read(paths['model_fit']);ml=read(paths['model_ledger'])
    if (any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [training,review]) or ledger['roots']['lifecycle_review']!=file_pin(paths['lifecycle_review']) or ml['fit']!=file_pin(paths['model_fit']) or fit['update']!=2000 or fit['training_running']):raise ValueError('Reviewed lifecycle/final model prerequisites required')
    contexts=context_pins(read(sp['plant_context_report']));actors={a['arm']:a['checkpoint'] for a in training['arms']}
    # Reconstruct the original ownership context,not an ad-hoc unit fit.
    fc=verify_file_pin(fit['config']);spec=fixed_spec(tomllib.loads(fc.read_text()));fpaths={k:(fc.parent/spec[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']};finputs={k:file_pin(p) for k,p in fpaths.items() if k not in ['registration','output']}
    common,old_fit,old,recovery,data,prep,path_data,products=data_view(fpaths);_,context=registered_context(spec,{**fpaths,'config':fc},finputs,common,old_fit,old,products);del data,prep,path_data,products
    historical=read(ROOT/'runs/pen_shared_snapshot_transport_v1/report.json');lifecycle=read(verify_file_pin(review['run']));records=[r['transport'] for r in historical['rows']]+[c['transport'] for a in lifecycle['arms'] for c in a['captures']];max_delta=max(r['payload']['compressed']['bytes'] for r in records);max_static=max(r['shared']['compressed']['bytes'] for r in records);projected=2*max_delta*57408+2*max_static*54+5*2**30;reserve=32*2**30
    storage=dict(max_sample_delta_bytes=max_delta,max_sample_static_bytes=max_static,projected_bytes=projected,reserve_bytes=reserve,free_bytes_at_registration=shutil.disk_usage(ROOT).free)
    inputs={k:file_pin(p) for k,p in {**sp,**{k:paths[k] for k in keys if k not in ['registration','output']}}.items()};inputs.update(model_checkpoint=fit['checkpoint'],shared_transport=file_pin(ROOT/'runs/pen_shared_snapshot_transport_v1/report.json'))
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/run_pen_backend_control_study.py',ROOT/'tools/evaluate_pen_long_control.py']
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs') or projected+reserve>=storage['free_bytes_at_registration']:raise ValueError('Fresh full54 registration with storage reserve required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_native_screened_rollout_registration:1',config=file_pin(cp),inputs=inputs,actors=actors,contexts=contexts,implementation_pins=implementation_pins(),source_receipt=file_pin(out/'source_receipt.json'),cases=cases,storage=storage,native_budget_max=300288,optimizer_steps=0));print(json.dumps(dict(registration=file_pin(paths['registration']),storage_GiB=projected/2**30,cases=len(cases))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['actors']!=actors or reg['contexts']!=contexts or reg['cases']!=cases or reg['implementation_pins']!=implementation_pins():raise ValueError('Registered full54 inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True);out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs') or projected+reserve>=shutil.disk_usage(ROOT).free:raise ValueError('Fresh full54 output and free-space reserve required')
    out.mkdir();receipt=freeze(out);write_json(out/'intent.json',dict(cases=cases,reference_controls_per_case=368,storage=storage,native_budget_max=300288));counter=WorkCounter();packets=runtime=None;completed=[];stage='models';monitor=GPUMonitor();models={};initial_models={}
    from mjlab.sim.sim import Simulation
    original=Simulation.step
    def step(sim):
        if packets is not None and packets.recorder.enabled:packets.recorder.before_step()
        result=counter.native_call(original,sim)
        if packets is not None and packets.recorder.enabled:packets.recorder.after_step()
        return result
    Simulation.step=step
    def guard():
        free=shutil.disk_usage(ROOT).free
        if free<reserve+max(4*max_delta,64*2**20):raise ValueError('Rolling32GiB storage reserve reached;partial experiment retained')
        return free
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0');checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
        for name in design['models']:
            saved=checkpoint['arms'][name]
            if saved['update']!=2000:raise ValueError('Every final2000 model required')
            owned=restore_arm(saved,context,device);model=owned['model'];model.eval();model.requires_grad_(False);models[name]=model;initial_models[name]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};del owned
        monitor.start()
        with backend_scope(sp['backend_manifest']) as backend,torch.jit.optimized_execution(False):
            try:
                for declared in cases:
                    case_start=time.perf_counter();stage=declared['name'];guard();folder=out/stage;folder.mkdir();state=torch.load(verify_file_pin(actors[declared['teacher']]),map_location='cpu',weights_only=False);runtime=BackendSupportPenRuntime(sp['objective_config'],1);actor,admission=actor_initializer(runtime,state,actors[declared['teacher']],contract=contract,plant_pin=contexts[declared['teacher']],expected_iteration=2000);before_actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};packets=LivePenPackets(runtime,receipt,counter,folder,'FULL54_NATIVE_SCREENED_OFFLINE_NEVER_FIT');scope=FullRehashVerifierScope(cfg['verifier'],reg['implementation_pins']);nodes=[];query_count=0;nn_wall=0.;actor_wall=0.;stop_reason=None
                    with scope:
                        for index in range(368):
                            guard();nf=folder/('P'+str(index+1).zfill(3));nf.mkdir();t=time.perf_counter()
                            with torch.no_grad():raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy()
                            actor_wall+=time.perf_counter()-t;source,sourcepin=packets.capture(nf,'source');phase=int(runtime.motion.time_steps[0]);stage=declared['name']+'_P'+str(phase).zfill(3);decision=from_owned_snapshot(source,phase,packets.template,raw);prop=None
                            if declared['controller']!='UNMODIFIED_ACTOR':
                                direction=np.zeros(20);finite=True;t=time.perf_counter()
                                if declared['controller']=='WM':
                                    model=models[declared['model']];ctx=batch_context({'live':decision['controller_context']},['live'],device);g=gradients(model,torch.as_tensor(decision['input102'][None],device=device),torch.as_tensor(raw[None],device=device),ctx,packets.reward_cfg,runtime.objective_cfg);ga={k:v.cpu().numpy() for k,v in g.items()};mode_index=0 if declared['score_mode']=='OBSERVABLE' else 2;direction=ga['RAW'][0,mode_index,declared['teacher_index']];finite=bool(np.isfinite(ga['values']).all());np.savez_compressed(nf/'gradient.npz',**ga)
                                    if not np.array_equal(ga['computed_PD'][0],decision['actual_PD20']):raise ValueError('WM controller value differs from actual original PD')
                                elif declared['controller']=='NATIVE_RANDOM':direction=random_direction(9901,declared['repetition'],declared['teacher_index'],phase)
                                prop=proposal(raw,direction,prediction_finite=finite);nn_wall+=time.perf_counter()-t;np.savez_compressed(nf/'proposal.npz',raw=prop['raw'],direction=direction,actual_raw_delta=prop['actual_raw_delta']);write_json(nf/'proposal.json',{k:v for k,v in prop.items() if k not in ['raw','actual_raw_delta']})
                            base,base_end=packets.query(raw,nf/'BASE',stage+'_BASE',source,sourcepin);query_count+=1;alt=None
                            if not base['physical_audit']['valid']:selection=dict(selected='BASE',accepted=False,stop=True,reason='BASE_PHYSICAL_INVALID');stop_reason='BASE_PHYSICAL_INVALID'
                            elif declared['controller']=='UNMODIFIED_ACTOR':selection=dict(selected='BASE',accepted=False,stop=base['terminated'] or base['truncated'],reason='UNMODIFIED_ACTOR')
                            else:
                                guard();packets.restore(sourcepin,stage+'_source');alt,alt_end=packets.query(prop['raw'],nf/'ALT',stage+'_ALT',source,sourcepin);query_count+=1;selection=choose(outcome(base),outcome(alt))
                                if selection['selected']=='BASE':packets.restore(base['end'],stage+'_end_BASE')
                            selected=alt if selection['selected']=='ALT' else base
                            node=dict(phase=phase,BASE=file_pin(nf/'BASE/report.json'),ALT=file_pin(nf/'ALT/report.json') if alt is not None else None,ALT_missing_reason='UNMODIFIED_ACTOR' if declared['controller']=='UNMODIFIED_ACTOR' else ('BASE_PHYSICAL_INVALID' if alt is None else None),selection=selection,selected_reward=selected['source_reward'],selected_terminated=selected['terminated'],selected_truncated=selected['truncated'],free_bytes_after=guard())
                            nodes.append(node);write_json(nf/'selection.json',node);write_json(out/'work.json',dict(stage=stage,completed_cases=len(completed),selected_controls=len(nodes),elapsed_wall_s=time.perf_counter()-started,**counter.metadata()))
                            if selection['stop']:
                                if stop_reason is None:stop_reason='REFERENCE_COMPLETION' if selected['truncated'] else 'TASK_FAILURE'
                                break
                            if int(runtime.motion.time_steps[0])!=phase+1 or bool(runtime.pending[0]):raise ValueError('Selected continuation phase/pending contract differs')
                    actor_equal=raw_tree_equal(before_actor,actor.state_dict());result=dict(declared,admission=admission,nodes=nodes,selected_controls=len(nodes),selected_duration_s=.02*len(nodes),missing_controls=list(range(len(nodes)+1,369)),native_queries=query_count,accepted_proposals=sum(n['selection']['accepted'] for n in nodes),stop_reason=stop_reason,
                        full_task_success=stop_reason=='REFERENCE_COMPLETION' and len(nodes)==368,restores=packets.restores,captures=packets.captures,actor_unchanged=actor_equal,scope=scope.audit(),wall_s=time.perf_counter()-case_start,actor_wall_s=actor_wall,proposal_wall_s=nn_wall)
                    write_json(folder/'report.json',result);completed.append(dict(name=declared['name'],report=file_pin(folder/'report.json'),selected_controls=len(nodes),native_queries=query_count,accepted_proposals=result['accepted_proposals'],stop_reason=stop_reason,full_task_success=result['full_task_success'],actor_unchanged=actor_equal,scope=result['scope'],wall_s=result['wall_s']));runtime.close();runtime=packets=None
                    if not actor_equal:raise ValueError('Frozen actor/normalizer changed')
                checks=dict(all54_declared=len(completed)==len(cases)==54,all_case_order=[c['name'] for c in completed]==[c['name'] for c in cases],actual_native_exact=counter.native==counter.native_attempted==8*sum(c['native_queries'] for c in completed)<=300288,no_optimizer=counter.optimizer==counter.optimizer_attempted==0,
                    all_actor_normalizers=all(c['actor_unchanged'] for c in completed),all_WM_buffers_weights=all(raw_tree_equal(initial_models[n],m.state_dict()) for n,m in models.items()),all_scopes=all(c['scope']['validated'] and c['scope']['bindings_restored'] for c in completed),all_outcomes_recorded=all(c['stop_reason'] in ['REFERENCE_COMPLETION','TASK_FAILURE','BASE_PHYSICAL_INVALID'] for c in completed))
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True);monitor.stop();write_json(out/'gpu.json',dict(samples=monitor.samples,errors=monitor.errors))
                result=dict(schema='pen_full54_native_screened_rollout:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),cases=completed,gpu=file_pin(out/'gpu.json'),**counter.metadata(),normalizer_fits=0,wall_s=time.perf_counter()-started,scope='All54 declared offline native-screened outcomes,not independent seeds/ego/hardware or imagined-Actor A1/A2. Physical invalidity and task failures remain failures;workflow checks do not imply policy success. Independent saved review still required.')
                write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],cases=len(completed),**counter.metadata())),flush=True)
                if not all(checks.values()):raise ValueError('Full54 workflow discrepancy retained')
            except Exception as e:
                write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,completed=completed,missing_cases=[c for c in cases if c['name'] not in {x['name'] for x in completed}],**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=packets=None
    except Exception as e:
        if not (out/'failure.json').exists():write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,completed=completed,**counter.metadata(),resumable=False))
        raise
    finally:
        if monitor.thread.is_alive():monitor.stop()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
