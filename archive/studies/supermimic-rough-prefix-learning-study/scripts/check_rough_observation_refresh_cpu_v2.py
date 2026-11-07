"""Real installed ObservationManager/CircularBuffer/DelayBuffer CPU contracts.

Env/Entity state writes are narrow CPU fixtures. No native Env, physics, CUDA
device or optimizer is constructed. Do not generalize to private camera models.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,ast,copy,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix import runtime_v2 as rt
from rough_prefix.bundle_v2 import load_bundle
from mjlab.managers.observation_manager import ObservationManager,ObservationGroupCfg,ObservationTermCfg
from mjlab.utils.buffers import CircularBuffer,DelayBuffer
spec=importlib.util.spec_from_file_location('rough_retained_fixture_definitions',ROOT/'scripts/check_rough_runtime_cpu.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def signal(env):return torch.cat((env.sim.data.qvel[:,6:26],env.sim.data.ctrl,env.motion.time_steps.float()[:,None]),-1)
def instant(env):return signal(env)[:,:2]


def buffer_snapshot(manager):
    out={}
    for group,values in manager._group_obs_term_history_buffer.items():
        for term,b in values.items():out['history/'+group+'/'+term]={'data':b._buffer.clone() if b.is_initialized else None,'pointer':b._pointer,'pushes':b._num_pushes.clone()}
    for group,values in manager._group_obs_term_delay_buffer.items():
        for term,b in values.items():out['delay/'+group+'/'+term]={'data':b._buffer._buffer.clone() if b.is_initialized else None,'pointer':b._buffer._pointer,'pushes':b._buffer._num_pushes.clone(),'lags':b._current_lags.clone(),'steps':b._step_count.clone(),'phase_offsets':b._phase_offsets.clone()}
    return out


class RealObsEnv(fixture.StubEnv):
    def __init__(self,cfg):
        super().__init__(cfg)
        groups={'policy':ObservationGroupCfg(terms={'history':ObservationTermCfg(func=signal,history_length=3,flatten_history_dim=False),'instant':ObservationTermCfg(func=instant)},concatenate_terms=False,enable_corruption=False),'critic':ObservationGroupCfg(terms={'delayed':ObservationTermCfg(func=signal,delay_min_lag=0,delay_max_lag=3,delay_update_period=3,delay_per_env_phase=True,delay_hold_prob=.25,history_length=2,flatten_history_dim=False)},concatenate_terms=False,enable_corruption=False)}
        self.observation_manager=ObservationManager(groups,self)
    def reset(self,*,seed=None,env_ids=None,options=None):
        # Installed Manager.reset ordering, with CPU state-writing fixtures.
        self.motion.prepare_reset(env_ids);self.action_manager.reset(env_ids)
        self.observation_manager.reset(env_ids=env_ids)
        self.motion.time_steps[env_ids]+=1
        obs=self.observation_manager.compute(update_history=True,env_ids=env_ids)
        self.native_reset_cache=obs;self.native_reset_buffers=buffer_snapshot(self.observation_manager);self.native_reset_rng=torch.get_rng_state().clone()
        return obs,{}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'runtime_v2_source.py').write_bytes(Path(rt.__file__).read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes());b=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');native=json.loads(Path(b['spec']['source_pins']['native_manifest']['path']).read_text());saved=Path(native['native']['path']).parents[2]/'native_reader_loaded_CPU.npz'
    with np.load(saved,allow_pickle=False) as z:fixture.LOADED={k:z[k].copy() for k in z.files}
    checks=[];snapshots={}
    def check(name,call):
        try:
            if not call():raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    def cfg():return NS(auto_reset=False,events={'reset_to_reference_state':NS(mode='reset')},terminations={'reference_end':NS(func=lambda env:None,time_out=True)})
    original_reset=RealObsEnv.reset
    torch.manual_seed(97)
    with rt.rough_runtime(RealObsEnv,fixture.Motion,fixture.StubMDP,fixture.ScopeRunner,b['arrays'],b['descriptor']):
        env=RealObsEnv(cfg());obs,_=env.reset();manager=env.observation_manager;after=buffer_snapshot(manager)
        check('actual_installed_ObservationManager_constructed_CPU',lambda:type(manager) is ObservationManager and all(isinstance(bb,CircularBuffer) for values in manager._group_obs_term_history_buffer.values() for bb in values.values()) and all(isinstance(bb,DelayBuffer) for values in manager._group_obs_term_delay_buffer.values() for bb in values.values()))
        check('fullreset_refreshed_cache_replaced',lambda:manager._obs_buffer is obs and manager._obs_buffer is not env.native_reset_cache)
        check('fullreset_instant_observation_has_corrected_analytic_state',lambda:torch.equal(obs['policy']['instant'],instant(env)))
        expected=signal(env)
        check('fullreset_all_history_slots_backfilled_corrected_state',lambda:torch.equal(obs['policy']['history'],expected[:,None].expand(-1,3,-1)))
        check('fullreset_delayed_history_corrected_after_native_first_init',lambda:torch.equal(obs['critic']['delayed'],expected[:,None].expand(-1,2,-1)))
        check('correction_does_not_consume_extra_RNG',lambda:torch.equal(torch.get_rng_state(),env.native_reset_rng))
        check('correction_does_not_advance_global_pointers',lambda:all(after[k]['pointer']==v['pointer'] for k,v in env.native_reset_buffers.items()))
        check('correction_preserves_native_delay_lags_schedule_offsets',lambda:all(torch.equal(after[k][field],v[field]) for k,v in env.native_reset_buffers.items() if k.startswith('delay/') for field in ('lags','steps','phase_offsets')))
        rejected=False
        try:manager.compute(update_history=False,env_ids=torch.tensor([1]))
        except AssertionError:rejected=True
        check('installed_API_reproduces_v1_false_with_ids_assertion',lambda:rejected)
        cached=manager._obs_buffer;env.sim.data.qvel[1,6]=999.;stale=manager.compute(update_history=False)
        check('dropping_ids_false_would_return_stale_cache',lambda:stale is cached and not torch.equal(stale['policy']['instant'][1],instant(env)[1]))
        # Mature timelines, including stochastic delayed terms, before a partial
        # native reset. No dynamics; only controlled CPU observation inputs.
        for k in range(5):env.sim.data.qvel[:,6:26]+=.03;env.motion.time_steps+=1;manager.compute(update_history=True)
        before=buffer_snapshot(manager);untouched_obs=copy.deepcopy(manager._obs_buffer);ids=torch.tensor([1,3]);others=torch.tensor([0,2]);obs,_=env.reset(env_ids=ids);after=buffer_snapshot(manager);expected=signal(env)
        check('partialreset_selected_instant_corrected',lambda:torch.equal(obs['policy']['instant'][ids],instant(env)[ids]))
        check('partialreset_selected_history_replaces_stale_native_frame',lambda:torch.equal(obs['policy']['history'][ids],expected[ids,None].expand(-1,3,-1)) and torch.equal(obs['critic']['delayed'][ids],expected[ids,None].expand(-1,2,-1)))
        check('partialreset_nonselected_history_and_delay_storage_unchanged',lambda:all(torch.equal(after[k]['data'][:,others],before[k]['data'][:,others]) for k in before))
        check('partialreset_nonselected_push_counts_unchanged',lambda:all(torch.equal(after[k]['pushes'][others],before[k]['pushes'][others]) for k in before))
        check('partialreset_global_pointers_unchanged',lambda:all(after[k]['pointer']==before[k]['pointer'] for k in before))
        check('partialreset_nonselected_lags_steps_phaseoffsets_unchanged',lambda:all(torch.equal(after[k][field][others],before[k][field][others]) for k in before if k.startswith('delay/') for field in ('lags','steps','phase_offsets')))
        check('partialreset_correction_no_second_lag_phase_draw',lambda:torch.equal(torch.get_rng_state(),env.native_reset_rng))
        check('partialreset_nonselected_output_history_not_advanced',lambda:torch.equal(obs['policy']['history'][others],untouched_obs['policy']['history'][others]) and torch.equal(obs['critic']['delayed'][others],untouched_obs['critic']['delayed'][others]))
        check('partialreset_selected_frame_push_count1',lambda:all(bool((v['pushes'][ids]==1).all()) for v in after.values()))
        snapshots['partial_after']={k:{field:(v.tolist() if torch.is_tensor(v) else v) for field,v in value.items() if field!='data'} for k,value in after.items()}
    check('runtime_v2_context_restores_realobs_Env_method',lambda:RealObsEnv.reset is original_reset)
    # Direct installed buffer contracts, independent of the manager wrapper.
    circular=CircularBuffer(3,4,'cpu');circular.append(torch.arange(4)[:,None].float());circular.append(torch.arange(4)[:,None].float()+1);old=circular._buffer.clone();pointer=circular._pointer;circular.backfill(torch.full((4,1),42.),torch.tensor([1,3]))
    check('real_CircularBuffer_backfill_no_pointer_or_otherrow_advance',lambda:circular._pointer==pointer and torch.equal(circular._buffer[:,[0,2]],old[:,[0,2]]) and torch.equal(circular.buffer[[1,3]],torch.full((2,3,1),42.)))
    delay=DelayBuffer(0,3,4,'cpu',update_period=3,per_env_phase=True,hold_prob=.4);delay.append(torch.arange(4)[:,None].float());delay.compute();old=copy.deepcopy(delay.__dict__);rng=torch.get_rng_state();delay.backfill(torch.full((4,1),77.),torch.tensor([1,3]));delay.peek()
    check('real_DelayBuffer_backfill_peek_no_lag_schedule_RNG_advance',lambda:torch.equal(rng,torch.get_rng_state()) and all(torch.equal(getattr(delay,k),old[k]) for k in ('_current_lags','_step_count','_phase_offsets')) and delay._buffer._pointer==old['_buffer']._pointer)
    before=(ROOT/'src/rough_prefix/runtime.py').read_text();after_source=Path(rt.__file__).read_text();(out/'runtime_v1_to_v2.diff').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),after_source.splitlines(keepends=True),fromfile='runtime.py5736',tofile='runtime_v2.py')))
    def ast_without_reset(text):
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='reset' and any(isinstance(n,ast.Attribute) and n.attr=='observation_manager' for n in ast.walk(node)):
                for call in ast.walk(node):
                    if isinstance(call,ast.Call) and ast.unparse(call.func)=='self.observation_manager.compute':
                        for keyword in call.keywords:
                            if keyword.arg=='update_history':keyword.value=ast.Constant(value='NORMALIZED_BOOL_FOR_DIFF')
        return ast.dump(tree,include_attributes=False)
    check('only_executable_change_is_refresh_history_flag',lambda:ast_without_reset(before)==ast_without_reset(after_source))
    check('old5736source_and40CPUproof_retained',lambda:sha(ROOT/'src/rough_prefix/runtime.py')=='5736dd1fd678a39f87867461d30d0710e6242af10e42f6fa2d57f6c9301b1ff9' and (ROOT/'outputs/runtime_cpu_contract_v1/report.json').is_file() and (ROOT/'outputs/runtime_scope_cpu_v1/report.json').is_file())
    check('no_native_Env_physics_optimizer_or_CUDA',lambda:not torch.cuda.is_initialized())
    sources={k:pin(p) for k,p in {'runtime_v2':Path(rt.__file__),'checker':Path(__file__),'ObservationManager':Path(sys.modules[ObservationManager.__module__].__file__),'CircularBuffer':Path(sys.modules[CircularBuffer.__module__].__file__),'DelayBuffer':Path(sys.modules[DelayBuffer.__module__].__file__)}.items()}
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_pins':sources,'snapshots':snapshots,'scope':'Real installed ObservationManager/CircularBuffer/DelayBuffer construction and methods on CPU. Env/Entity physical writes remain narrow fixtures. Stateless terms only, matching actual play-mode terms; no claim for private camera/history models.','native_Env_constructed':False,'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'actual_optimizer_steps':0};(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='snapshots'},indent=2));raise SystemExit(result['status']!='PASS')
if __name__=='__main__':main()
