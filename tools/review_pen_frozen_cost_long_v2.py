"""Saved continuous source/Actor/physics/reward/trajectory review;no replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import settings,schedule,read
from supermimic_research.strict_json import write_json


def main(a):
    started=time.perf_counter(); cp=a.config.resolve(); cfg,paths,inputs,actors,contexts,q=settings(cp); reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['cases']!=schedule() or reg['actors']!=actors: raise ValueError('Exact registered long evaluation required')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    def freeze(folder):
        source=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']],folder/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(folder/'source_receipt.json',source); return source
    if a.register:
        from supermimic_research.runtime.pen_continuous_math_view_v1 import terminal_reason
        from supermimic_research.runtime.shared_snapshot_transport import validate
        try: validate(dict(schema='pen_continuous_evaluation_numeric_view:1'))
        except ValueError: rejected=True
        else: rejected=False
        checks=dict(all18=len(schedule())==18,cap52992=18*368*8==52992,reduced_view_not_checkpoint=rejected,
            physical_censor=terminal_reason(False,True,False,4)=='PHYSICAL_AUDIT_STOP',task_stop=terminal_reason(True,True,False,4)=='TASK_FAILURE',reference=terminal_reason(True,False,True,368)=='REFERENCE_COMPLETION')
        folder=paths['reader_registration'].parent
        if folder.exists() or not all(checks.values()): raise ValueError('Fresh successful reader registration required')
        folder.mkdir(); freeze(folder); write_json(paths['reader_registration'],dict(schema='pen_frozen_cost_long_reader_registration:1',worker_registration=file_pin(paths['registration']),config=file_pin(cp),source_receipt=file_pin(folder/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,observation_scaled_L2_bound=1e-5,RAW_scaled_L2_bound=1e-4))
        print(json.dumps(dict(registration=file_pin(paths['reader_registration']),checks=[sum(checks.values()),len(checks)],native=0)),flush=True); return
    rr=read(paths['reader_registration']); producer=read(paths['output']/'report.json')
    if rr['worker_registration']!=file_pin(paths['registration']) or producer['passed']!=producer['total'] or not all(producer['checks'].values()) or (paths['output']/'failure.json').exists(): raise ValueError('Closed successful18-case producer required')
    if producer['registration']!=file_pin(paths['registration']) or producer['reader_registration']!=file_pin(paths['reader_registration']): raise ValueError('Producer registration ownership changed')
    for item in [rr,producer]: verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
    out=paths['reader_output']
    if out.exists(): raise ValueError('Fresh saved review output required')
    import numpy as np
    import torch
    import mujoco
    from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
    from supermimic_research.learning.world_model import raw_tree_equal
    from supermimic_research.runtime.shared_snapshot_transport import load_shared
    from supermimic_research.runtime.pen_continuous_math_view_v1 import context,physics_view,policy,terminal_reason
    from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import augmented_trace,near_error,static_math_fields
    from supermimic_research.runtime.controlled_evaluation import summarize_trace
    from supermimic_research.mechanics.probe_row_review import review_recorded_row
    from supermimic_research.mechanics.pen_reward_chain import controller,native_components,ACTION,MOTION,A
    torch.set_num_threads(4); gates=tomllib.loads(paths['gates'].read_text()); objective=tomllib.loads(paths['objective_config'].read_text()); adapter=(paths['objective_config'].parent/objective['adapter_config']).resolve(); reward_cfg=tomllib.loads(adapter.read_text())['reward']
    out.mkdir(); receipt=freeze(out); all_checks={}; case_rows=[]; total_controls=0; model_cache={}
    with torch.no_grad(),torch.jit.optimized_execution(False),patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved review cannot step')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved review cannot forward')):
        for slot,decl in zip(producer['rows'],reg['cases']):
            c=read(verify_file_pin(slot['report'])); folder=Path(slot['report']['path']).parent; capsule=[file_pin(p) for p in sorted(folder.rglob('*')) if p.is_file()]
            initial=load_shared(c['initial']); end=load_shared(c['end']); m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model']))); physical=dict(np.load(verify_file_pin(c['physical_context'])))
            label=decl['policy']
            if label not in model_cache:
                state=torch.load(verify_file_pin(actors[label]),map_location='cpu',weights_only=False); conf=state['config']; actor=GaussianPPOActor(161,20,init_std=conf['ppo_init_std'],std_range=(conf['ppo_std_min'],conf['ppo_std_max'])); actor.load_state_dict(state['modules']['actor'],strict=True); actor.eval(); actor.requires_grad_(False); model_cache[label]=(actor,state)
            actor,state=model_cache[label]; after=torch.load(verify_file_pin(c['actor_after']),map_location='cpu',weights_only=False)
            profile=read(folder/'profile.json'); verifier=read(verify_file_pin(slot['verifier']))
            checks=dict(declaration=all(c[k]==v for k,v in decl.items()),target_backend=initial['metadata']['termination_backend']==end['metadata']['termination_backend']==reg['identities'][decl['algorithm']],
                initial_phase=int(initial['python']['tensors'][MOTION+(A('time_steps'),)][0])==1,
                source_model=raw_tree_equal(initial['native']['model'],end['native']['model']),static_inputs=raw_tree_equal(static_math_fields(initial),static_math_fields(end)),
                frozen_actor=after.keys()==state['modules']['actor'].keys() and all(torch.equal(v,state['modules']['actor'][k]) for k,v in after.items()),
                profile=profile['contract']==reg['identities'][decl['algorithm']] and profile['entered'] and profile['stopping_unchanged'] and profile['bindings_restored'],
                verifier=verifier['validated'] and verifier['bindings_restored'] and not verifier['source_epoch_used'])
            rows=[]; traces=[]; previous=None; last_raw=initial['python']['tensors'][ACTION+(A('_raw_actions'),)][0].numpy(); last_ema=initial['python']['tensors'][ACTION+(A('_processed_actions'),)][0].numpy()
            for index,pin in enumerate(c['controls']):
                control=read(verify_file_pin(pin)); sample=torch.load(verify_file_pin(control['control_source']),map_location='cpu',weights_only=False); dynamic=sample['dynamic']; z=dict(np.load(verify_file_pin(control['trace']))); w=dict(np.load(verify_file_pin(control['window'])))
                cpu_obs=policy(initial,dynamic,c); obs=sample['policy_observation']; obs_error=near_error(cpu_obs.numpy(),obs); raw=actor.distribution(cpu_obs,validate_args=False).mean.numpy()[0]; raw_error=near_error(raw,sample['sampled_mean_RAW'])
                ctx=context(initial,dynamic,c); predicted,ema,applied=controller(torch.tensor(sample['sampled_mean_RAW'][None]),ctx)
                _,audit=review_recorded_row(control,z,w,physics_view(initial,dynamic,c),m,physical,dict(role=c['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001)); values,flags,errors=native_components(control,z,w,ctx,reward_cfg,objective)
                reason=terminal_reason(audit['valid'],control['terminated'],control['truncated'],index+1)
                local=dict(phase=control['phase']==dynamic['phase']==index+1,observation=obs_error is not None and obs_error<=1e-5,RAW=raw_error is not None and raw_error<=1e-4,
                    source_chain=previous is None or (np.array_equal(previous['post_qpos'][-1],dynamic['arrays']['qpos'][0]) and np.array_equal(previous['post_qvel'][-1],dynamic['arrays']['qvel'][0]) and np.array_equal(previous['post_qacc_warmstart'][-1],dynamic['arrays']['qacc_warmstart'][0])),
                    RAW_feedback=np.array_equal(last_raw,dynamic['last_raw'].numpy()[0]),EMA_feedback=np.array_equal(last_ema,dynamic['EMA'].numpy()[0]),
                    physical_audit=audit==control['physical_audit'],reward_flags=flags==control['reward_checks'] and all(flags.values()),reward_errors=errors==control['reward_errors'],reward_value=abs(float(values[-1])-control['source_reward'])<=2e-6,
                    transmitted_PD=np.max(abs(predicted.numpy()[0]-np.asarray(control['actual_PD20'])))<=2e-7,stop=control['stop_reason']==reason,only_final_stop=(reason is not None)==(index==len(c['controls'])-1),producer_checks=control['passed']==control['total'] and all(control['checks'].values()))
                last_raw=applied.numpy()[0]; last_ema=ema.numpy()[0]; previous=z; traces.append(augmented_trace(z,m,c)); rows.append(dict(phase=index+1,checks={k:bool(v) for k,v in local.items()},physical_valid=audit['valid'],stop_reason=reason,observation_scaled_L2=obs_error,RAW_scaled_L2=raw_error))
                checks.update({f'P{index+1:03}_{k}':bool(v) for k,v in local.items()})
            saved=dict(np.load(verify_file_pin(c['trace']))); joined={k:np.concatenate([z[k] for z in traces]) for k in traces[0]}; refpose=initial['python']['arrays'][(A('support_table'),A('pose'))]
            ref=dict(time_s=np.arange(369)*.02,COM_world_m=refpose[:,:3],quaternion_wxyz=refpose[:,3:])
            summary=summarize_trace(joined,ref,.0025,rows[-1]['stop_reason']=='REFERENCE_COMPLETION' and len(rows)==368,gates)
            prefix=next((i for i,r in enumerate(rows) if not r['physical_valid']),len(rows)); reason=rows[-1]['stop_reason']
            checks.update(joined_trace=all(np.array_equal(v,saved[k]) for k,v in joined.items()),reference=all(np.array_equal(v,saved[k]) for k,v in ref.items()),strict_summary=summary==c['summary'],
                controls=c['selected_controls']==len(rows),native=c['native_integrations']==8*len(rows),missing_future=c['missing_controls']==list(range(len(rows)+1,369)),
                final_stop=reason==c['stop_reason'],valid_prefix=c['valid_prefix_controls']==prefix and c['valid_prefix_s']==.02*prefix,
                final_state=np.array_equal(joined['post_qpos'][-1],end['native']['data']['arrays']['qpos'][0]) and np.array_equal(joined['post_qvel'][-1],end['native']['data']['arrays']['qvel'][0]),
                full_success=c['full_task_success']==bool(summary['qualified'] and reason=='REFERENCE_COMPLETION' and len(rows)==368),actor_transfer=read(verify_file_pin(slot['actor_transfer']))['transfer']=='ACTOR_PARAMETERS_AND_FROZEN_NORMALIZATION_ONLY')
            for pin in capsule: verify_file_pin(pin)
            checks['input_file_inventory']=[p['path'] for p in capsule]==[str(p.resolve()) for p in sorted(folder.rglob('*')) if p.is_file()]
            target=out/c['name']; target.mkdir(); result=dict(schema='pen_frozen_cost_long_case_saved_review:1',producer_case=slot['report'],checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),controls=rows,
                input_file_capsule=capsule,summary=summary,stop_reason=reason,valid_prefix_s=.02*prefix,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0)
            write_json(target/'report.json',result); case_rows.append(dict(name=c['name'],report=file_pin(target/'report.json'))); all_checks[c['name']]=all(checks.values()); total_controls+=len(rows)
        all_checks.update(all18=len(case_rows)==18,exact_order=[r['name'] for r in case_rows]==[d['name'] for d in reg['cases']],native_exact=producer['actual_native_integrations']==producer['attempted_native_integrations']==8*total_controls<=52992)
    verify_source_snapshot(receipt,live=True); result=dict(schema='pen_frozen_cost_long_saved_review:1',producer=file_pin(paths['output']/'report.json'),registration=file_pin(paths['reader_registration']),source_receipt=file_pin(out/'source_receipt.json'),rows=case_rows,
        checks=all_checks,passed=sum(all_checks.values()),total=len(all_checks),controls_recomputed=total_controls,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Separate saved Actor/source/RAW-feedback/physics/reward/strict-outcome recomputation. Reduced numeric views are not resume states. No native step/forward,fit or new policy selection.')
    write_json(out/'report.json',result); print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],controls=total_controls,native=0)),flush=True)
    if not all(all_checks.values()): raise ValueError('Saved long evaluation discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--register',action='store_true'); main(p.parse_args())
