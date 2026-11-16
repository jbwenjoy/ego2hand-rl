"""All-phase actual GPU feature parity and cache/mode ownership negative cases."""
import argparse,copy,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_native_policy import NativePenFeatures
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.gpu_model_snapshot import model_leaves


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def difference(left,right,tolerance):
    same_nan=torch.equal(torch.isnan(left),torch.isnan(right));finite=torch.isfinite(left)&torch.isfinite(right)
    same_nonfinite=torch.equal(torch.isfinite(left),torch.isfinite(right)) and torch.equal(left[torch.isinf(left)],right[torch.isinf(right)])
    error=float((left[finite]-right[finite]).abs().max()) if bool(finite.any()) else 0.
    exact=bool(((left==right)|(torch.isnan(left)&torch.isnan(right))).all())
    return dict(max_abs=error,exact=exact,nan_pattern=same_nan,nonfinite_pattern=same_nonfinite,passed=same_nan and same_nonfinite and error<=tolerance)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['num_envs']!=32:raise ValueError('Fresh actual GPU32 qualification')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt);runtime=None;rows=[];checks={}
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            runtime=SupportPenRuntime((cp.parent/cfg['objective_config']).resolve(),32);f=runtime.features;c=runtime.reference_cache;d=runtime.env.sim.data
            saved=dict(qpos=d.qpos.clone(),qvel=d.qvel.clone(),phase=f.motion.time_steps.clone(),raw=f.action._raw_actions.clone(),EMA=f.action._processed_actions.clone())
            def compare(name):
                live=c.policy(f);fast=c.policy(f,fixed_frames=True);lc=NativePenFeatures.critic(f,live);fc=NativePenFeatures.critic(f,fast,wrist_rotation_transpose=c.critic_WT)
                row=dict(name=name,policy=difference(live,fast,cfg['policy_critic_absolute_tolerance']),critic=difference(lc,fc,cfg['policy_critic_absolute_tolerance']),reference=difference(NativePenFeatures.reference_pose_world(f),c.reference_pose_world(f.motion.frame_ids),cfg['reference_absolute_tolerance']))
                rows.append(row);checks[name]=all(row[k]['passed'] for k in ['policy','critic','reference'])
            for phase in range(369):f.motion.time_steps.fill_(phase);compare(f'PHASE{phase:03d}')
            for phase in [-10,-1,0,368,369,1000]:f.motion.time_steps.fill_(phase);compare(f'CLAMP{phase}')
            f.motion.time_steps.copy_(torch.arange(32,device=runtime.device)*11);compare('MIXED_PHASES')
            d.qpos[:,:3]+=torch.linspace(-.01,.01,32,device=runtime.device)[:,None];d.qpos[:,f.qids]+=.025;d.qvel.add_(.031);f.action._raw_actions+=.07;f.action._processed_actions+=.04;compare('LIVE_DYNAMIC_SUFFIX')
            d.qpos[:,f.qa+3:f.qa+7]*=1.07;compare('NONUNIT_CURRENT_QUAT')
            d.qpos[0,f.qa+3:f.qa+7]=0;compare('ZERO_CURRENT_QUAT_NAN')
            d.qpos.copy_(saved['qpos']);d.qvel.copy_(saved['qvel']);f.motion.time_steps.copy_(saved['phase']);f.action._raw_actions.copy_(saved['raw']);f.action._processed_actions.copy_(saved['EMA'])
            checks['temporary_inputs_restored']=all(torch.equal(value,dict(qpos=d.qpos,qvel=d.qvel,phase=f.motion.time_steps,raw=f.action._raw_actions,EMA=f.action._processed_actions)[k]) for k,value in saved.items())
            for name in ['policy_WT','critic_WT']:
                matrix=getattr(c,name);old=matrix.clone();matrix[0,0,0]+=.001
                try:runtime.begin_collection();rejected=False
                except ValueError:rejected=True
                matrix.copy_(old);checks[name+'_corruption_rejected']=rejected;runtime.validate_resources();c.validate(f)
            old=c.critic_WT;c.critic_WT=old.clone()
            try:runtime.begin_collection();rejected=False
            except ValueError:rejected=True
            c.critic_WT=old;checks['matrix_allocation_replacement_rejected']=rejected
            runtime.begin_collection()
            try:runtime.set_fixed_frame_features(True);rejected=False
            except ValueError:rejected=True
            runtime.end_collection();checks['in_collection_mode_switch_rejected']=rejected
            f.fixed_frame_cache=True
            try:runtime.validate_resources();rejected=False
            except ValueError:rejected=True
            f.fixed_frame_cache=False;checks['unapproved_mode_mutation_rejected']=rejected
            runtime.set_fixed_frame_features(True);checks['cached_mode_public_observation']=runtime.metadata()['fixed_frame_features']=='CACHED' and runtime.observation()['policy'].shape==(32,161)
            state=native.capture(runtime,receipt);native.preflight(runtime,state);torch.save(state,out/'native_source.pt')
            path=next(p for p in state['python']['scalars'] if p and p[-1]==('attr','fixed_frame_cache'));bad=copy.deepcopy(state);bad['python']['scalars'][path]=False
            try:native.preflight(runtime,bad);rejected=False
            except ValueError as error:rejected='fixed-frame mode' in str(error)
            checks['saved_mode_mismatch_rejected_before_write']=rejected
            checks['saved_mode_rejection_entire_Model_Data_unchanged']=all(v.numpy().tobytes()==state['native'][key]['arrays'][name].tobytes() for key,obj in [('model',runtime.env.sim.wp_model),('data',runtime.env.sim.wp_data)] for name,v in model_leaves(obj)[0].items()) and f.fixed_frame_cache
            runtime.set_fixed_frame_features(False);runtime.validate_resources();c.validate(f);verify_source_snapshot(receipt,live=True)
            result=dict(schema='fixed_frame_GPU_qualification:1',config=file_pin(cp),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),
                all_feature_rows_exact=all(row[k]['exact'] for row in rows for k in ['policy','critic','reference']),max_abs={k:max(row[k]['max_abs'] for row in rows) for k in ['policy','critic','reference']},
                native_source=file_pin(out/'native_source.pt'),cache_metadata=c.metadata(),actual_native_integrations=0,optimizer_steps=0,scope='Actual CUDA32-world static feature qualification across369 phases/clamps/dynamic/NaN probes and ownership guards;not physical transition/TRAIN/task/gradient/ego evidence.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],exact=result['all_feature_rows_exact'],max_abs=result['max_abs'])),flush=True)
            if not all(checks.values()):raise ValueError('Retain actual GPU feature qualification discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),partial_rows=rows,checks=checks,actual_native_integrations=0,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
