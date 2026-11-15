"""One producer/restorer process for the owned native4 checkpoint contract."""
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
import warp as wp
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.runtime import owned_pen_state as owned
from supermimic_research.runtime import pen_portable_state as old
from supermimic_research.runtime.pen_native_policy import NativePenFeatures
from supermimic_research.mechanics.portable_contact_records import contacts


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def digest_state(state):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):h.update(old.tensor_bytes(x))
        elif isinstance(x,np.ndarray):h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    for k in ['python','native','rng']:add(state[k])
    return h.hexdigest()


def future(runtime):
    s=runtime.env.sim;d=s.wp_data
    return dict(qpos=d.qpos.numpy().copy(),qvel=d.qvel.numpy().copy(),ctrl=d.ctrl.numpy().copy(),time=d.time.numpy().copy(),
        contact=contacts(s.wp_model,d,s.mj_model,runtime.features.body),metrics=runtime.contact_metrics.numpy(),observation=old.tensor_bytes(runtime.observation()['policy']))


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve();W=a.worlds
    if out.exists() or not out.is_relative_to(ROOT/'runs') or W not in cfg['worlds']:raise ValueError('Fresh owned state worker')
    out.mkdir();ap=(cp.parent/cfg['adapter_config']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;phase='construction';checks={};rejections=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            runtime=OwnedPenRuntime(ap,W);checks['constructor_no_physics']=observed==0
            def capture():return owned.capture(runtime,receipt)
            def reject(label,state):
                before=digest_state(capture());reason=None
                try:owned.restore(runtime,state)
                except (ValueError,KeyError,TypeError) as error:reason=str(error)
                after=digest_state(capture());checks[label+'_rejected']=reason is not None;checks[label+'_unchanged']=before==after
                rejections.append(dict(name=label,reason=reason,before=before,after=after))
            ids=torch.tensor([0] if W==1 else [0,7,31],device='cuda:0',dtype=torch.int64)
            if a.mode=='producer':
                phase='prefix';runtime.begin_collection();policy_errors=[]
                for k in range(cfg['prefix_controls']):
                    w=torch.arange(W,device='cuda:0')[:,None];j=torch.arange(20,device='cuda:0')[None,:]
                    ob,_,term,trunc,_=runtime.step(.05*torch.sin(.2*k+.13*w+.03*j))
                    policy_errors.append(float((ob['policy']-NativePenFeatures.policy(runtime.features)).abs().max()))
                    done=(term|trunc).nonzero().flatten()
                    if len(done):runtime.reset(done)
                runtime.end_collection();checks['live_policy_exact']=max(policy_errors)==0
                control=capture();torch.save(control,out/'control_state.pt')
                checks['control_samples_eight']=bool(np.all(runtime.contact_metrics.numpy()['window']['counts'][:,0]==8))
                phase='midwindow';runtime.contact_metrics.reset_window()
                for _ in range(cfg['midwindow_native_samples']):runtime.env.sim.step()
                state=capture();checkpoint=out/'midwindow_state.pt';torch.save(state,checkpoint)
                saved=runtime.contact_metrics.numpy();checks['midwindow_samples_three']=bool(np.all(saved['window']['counts'][:,0]==3))
                runtime.contact_metrics.reset_window(ids)
                for _ in range(cfg['partial_followup_native_samples']):runtime.env.sim.step()
                after=runtime.contact_metrics.numpy()['window'];expected=np.full(W,5);expected[ids.cpu().numpy()]=2
                checks['partial_window_counts']=np.array_equal(after['counts'][:,0],expected)
                phase='restore_midwindow';before_integrations=observed;audit=owned.restore(runtime,torch.load(checkpoint,map_location='cpu',weights_only=False))
                checks['restore_no_physics']=before_integrations==observed
                checks['all_midwindow_bytes']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                checks['restored_window_exact']=all(np.array_equal(runtime.contact_metrics.numpy()['window'][k],v,equal_nan=True) for k,v in saved['window'].items())
                phase='negative_preflights';bad=copy.deepcopy(state);bad['schema']=old.SCHEMA;reject('old_schema',bad)
                bad=copy.deepcopy(state);bad['native']['data']['arrays']['qpos']=bad['native']['data']['arrays']['qpos'][:,:-1];reject('shape',bad)
                path=next(p for p in state['python']['tensors'] if p[:2]==(('attr','contact_views'),('item','counts')))
                bad=copy.deepcopy(state);bad['python']['tensors'][path][0,0]+=1;reject('alias',bad)
                path=next(p for p in state['immutable'] if p[-1]==('attr','prefix'))
                bad=copy.deepcopy(state);bad['python']['tensors'][path][0,0,0]+=1;reject('cache_bytes',bad)
                bad=copy.deepcopy(state);bad['python']['scalars'].pop(next(iter(bad['python']['scalars'])));reject('missing_scalar',bad)
                path=next(p for p in state['immutable_arrays'] if p[-1]==('attr','qids'))
                bad=copy.deepcopy(state);bad['python']['arrays'][path][0]=-1;reject('static_index',bad)
                path=next(p for p in state['immutable_scalars'] if ('attr','table_signature') in p)
                bad=copy.deepcopy(state);bad['python']['scalars'][path]='corrupt';reject('static_scalar',bad)
                path=(('attr','owned_native_integrations'),)
                bad=copy.deepcopy(state);bad['python']['scalars'][path]=float(bad['python']['scalars'][path]);reject('scalar_type',bad)
                bad=copy.deepcopy(state);bad['rng']['torch']=torch.zeros(1,dtype=torch.uint8);reject('RNG_format',bad)
                before=digest_state(capture());runtime._unknown_state=object()
                try:
                    try:owned.restore(runtime,state);checks['unknown_field_rejected']=False
                    except ValueError:checks['unknown_field_rejected']=True
                finally:del runtime._unknown_state
                checks['unknown_field_unchanged']=before==digest_state(capture())
                original_buffer=runtime.contact_metrics.window['Fn6_sum'];runtime.contact_metrics.window['Fn6_sum']=wp.zeros_like(original_buffer)
                try:
                    try:owned.restore(runtime,state);checks['pointer_replacement_rejected']=False
                    except ValueError:checks['pointer_replacement_rejected']=True
                finally:runtime.contact_metrics.window['Fn6_sum']=original_buffer
                checks['pointer_replacement_unchanged']=before==digest_state(capture())
                try:old.capture(runtime,receipt);checks['old_capture_rejected']=False
                except ValueError:checks['old_capture_rejected']=True
                checks['old_capture_unchanged']=before==digest_state(capture())
                # An invalid step must not erase metric accumulators.
                runtime.pending.fill_(True);metric_before=runtime.contact_metrics.numpy()
                try:runtime.step(torch.zeros(W,20,device='cuda:0'));checks['pending_step_rejected']=False
                except ValueError:checks['pending_step_rejected']=True
                runtime.pending.zero_();checks['pending_step_window_preserved']=all(np.array_equal(runtime.contact_metrics.numpy()['window'][k],v,equal_nan=True) for k,v in metric_before['window'].items())
                owned.restore(runtime,state)
            else:
                phase='fresh_restore';checkpoint=a.checkpoint.resolve();state=torch.load(checkpoint,map_location='cpu',weights_only=False);before_integrations=observed
                audit=owned.restore(runtime,state);checks['restore_no_physics']=observed==before_integrations==0
                checks['fresh_all_bytes']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                checks['fresh_window_three']=bool(np.all(runtime.contact_metrics.numpy()['window']['counts'][:,0]==3))
                torch.save(capture(),out/'restored_state.pt')
            phase='future_diagnostic'
            for _ in range(cfg['future_native_samples']):runtime.env.sim.step()
            torch.save(future(runtime),out/'future.pt');checks['future_metric_sample_count']=bool(np.all(runtime.contact_metrics.numpy()['window']['counts'][:,0]==4))
            # A CUDA-ID producer on another stream drives an actual partial env
            # reset; unselected worlds keep their dynamics and metric history.
            phase='cross_stream_partial_reset';d=runtime.env.sim.data;before_window=runtime.contact_metrics.numpy()['window']
            before_fields={k:getattr(d,k).clone() for k in ['qpos','qvel','ctrl','time']}
            before_phase=runtime.motion.time_steps.clone();before_EMA=runtime.action._processed_actions.clone()
            alternate=torch.cuda.Stream()
            alternate.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(alternate):runtime.reset(ids.clone())
            torch.cuda.current_stream().wait_stream(alternate)
            after_window=runtime.contact_metrics.numpy()['window'];selected=ids.cpu().numpy();other=np.setdiff1d(np.arange(W),selected)
            checks['partial_env_metric_reset']=bool(np.all(after_window['counts'][selected,0]==0))
            checks['partial_env_unselected_metrics']=all(np.array_equal(after_window[k][other],v[other],equal_nan=True) for k,v in before_window.items() if k!='global_error')
            checks['partial_env_unselected_native']=all(torch.equal(getattr(d,k)[other],v[other]) for k,v in before_fields.items())
            checks['partial_env_unselected_phase_EMA']=torch.equal(runtime.motion.time_steps[other],before_phase[other]) and torch.equal(runtime.action._processed_actions[other],before_EMA[other])
            runtime.reference_cache.validate(runtime.features)
            torch.save(capture(),out/'partial_reset_state.pt')
            expected=(cfg['prefix_controls']*8+cfg['midwindow_native_samples']+cfg['partial_followup_native_samples']+cfg['future_native_samples'])*W if a.mode=='producer' else W
            checks['actual_native_budget']=observed==expected
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='owned_pen_state_worker:1',config=file_pin(cp),mode=a.mode,worlds=W,source_receipt=file_pin(out/'source_receipt.json'),checkpoint=file_pin(checkpoint),future=file_pin(out/'future.pt'),
                audit=audit,checks=checks,passed=sum(checks.values()),total=len(checks),rejections=rejections,actual_native_integrations=observed,optimizer_steps=0,
                metadata=runtime.metadata(),scope=cfg['scope']+' Future record retained for independent comparison; immediate byte equality alone is not future physics qualification.')
            result['partial_reset_state']=file_pin(out/'partial_reset_state.pt')
            if a.mode=='producer':result['control_state']=file_pin(out/'control_state.pt')
            else:result['restored_state']=file_pin(out/'restored_state.pt')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],actual_native_integrations=observed)),flush=True)
            if not all(checks.values()):raise ValueError('Keep owned-state diagnostic failure')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,mode=a.mode,worlds=W,optimizer_steps=0));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worlds',type=int,required=True)
    p.add_argument('--mode',choices=['producer','restorer'],required=True);p.add_argument('--checkpoint',type=Path);main(p.parse_args())
