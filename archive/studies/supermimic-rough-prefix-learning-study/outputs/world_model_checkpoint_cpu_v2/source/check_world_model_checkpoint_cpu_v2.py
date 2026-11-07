"""Frozen CPU WorldModel consumer contracts and one forward/du per real final.

The diagnostic state is synthetic (proper identity rotation), not a raw-state
codec or observed physical transition. No Env, Warp or new physical query.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,sys,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'source';snap.mkdir();checks=[];methods={};errors=[]
    module_path=ROOT/'src/rough_prefix/world_model_checkpoint_v2.py'
    def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    source_pins={}
    for p in (Path(__file__),module_path,ROOT/'src/rough_prefix/checkpoint_v2.py'):
        (snap/p.name).write_bytes(p.read_bytes());source_pins[p.name]=pin(p)
    import numpy as np,torch
    torch.set_num_threads(2)
    from rough_prefix.bundle_v2 import load_bundle
    from rough_prefix.world_model_checkpoint_v2 import load_world_model_checkpoint,_physical_xi,_wm_payload
    from rough_prefix.checkpoint_v2 import _tensor_digest
    from rough_prefix.io import read,loadnp
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    source_pins['native_networks']=pin(sys.modules[WorldModel.__module__].__file__)
    b=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json')
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    def rejected(name,fn):
        try:fn()
        except (ValueError,KeyError,RuntimeError,TypeError) as exc:check(name,True,repr(exc))
        else:check(name,False)
    def rngs():return torch.get_rng_state().clone(),np.random.get_state(),random.getstate()
    def rng_equal(before):
        return torch.equal(before[0],torch.get_rng_state()) and np.array_equal(before[1][1],np.random.get_state()[1]) and before[1][0]==np.random.get_state()[0] and before[1][2:]==np.random.get_state()[2:] and before[2]==random.getstate()
    actual_checkpoints={method:ROOT/'outputs'/('primary_'+method.lower()+'_seed0_v1')/'train/model_final.pt' for method in ('A1L','P_ONLY','A2')}
    for method,path in actual_checkpoints.items():
        try:
            run=path.parents[1];health=ROOT/'outputs'/('health_primary_'+method.lower()+'_final_v1')/'report.json';folder=out/method.lower();folder.mkdir()
            bytes_before=pin(path);stat_before=path.stat();context_before=pin(run/'fixed_physics_context.npz');rng_before=rngs()
            wrapper,receipt=load_world_model_checkpoint(path,b,method,run,health)
            check(method+'_load_preserves_Torch_numpy_python_RNG',rng_equal(rng_before))
            check(method+'_actual_final2000_and_xi46_task368_separate',receipt['iteration']==2000 and receipt['physical_xi_dim']==46 and receipt['native_task_context_dim']==368 and receipt['physical_context_worlds_bitexact']==128)
            check(method+'_native_WM_parameters_frozen_CPUAMPfalse',isinstance(wrapper.wm,WorldModel) and not wrapper.training and not wrapper.wm.training and wrapper.wm.amp is False and all(not p.requires_grad for p in wrapper.parameters()))
            before={k:v.detach().clone() for k,v in wrapper.wm.state_dict().items()};before_digest=_tensor_digest(before);host_counts=(wrapper.wm.norm._host_count,wrapper.wm.delta_norm._host_count)
            x=torch.zeros((1,58),dtype=torch.float32);x[:,:20]=torch.from_numpy(b['arrays']['reference']['q_goal_rad'][0].astype(np.float32));x[:,43:52]=torch.eye(3).reshape(1,9)
            u=torch.from_numpy(b['baseline_u15'][0:1].copy()).requires_grad_(True);x_before=x.clone();u_before=u.detach().clone();observed=[]
            hook=wrapper.wm.register_forward_pre_hook(lambda mod,args:observed.append(tuple(value.detach().clone() for value in args)))
            rng_before=rngs()
            try:
                x1,contact=wrapper(x,u)
                loss=(x1*torch.linspace(.1,1.,58)[None]).sum()+.01*contact.sum()
                du,=torch.autograd.grad(loss,u)
            finally:hook.remove()
            check(method+'_one_forward_native_shapes_and_finite',x1.shape==(1,58) and contact.shape==(1,15) and bool(torch.isfinite(x1).all()) and bool(torch.isfinite(contact).all()) and len(observed)==1)
            captured=observed[0]
            check(method+'_native_dynamics_exact_x58_xi46_zero174',captured[2].shape==(1,278) and torch.equal(captured[0],x) and torch.equal(captured[1],u_before) and torch.equal(captured[2][:,:58],x) and torch.equal(captured[2][:,58:104],wrapper.physical_xi[None]) and bool((captured[2][:,104:]==0).all()) and wrapper.wm.inputs(captured[2],captured[1]).shape==(1,298))
            check(method+'_input_autograd_available_finite_no_parameter_grads',x1.requires_grad and contact.requires_grad and du.shape==(1,20) and bool(torch.isfinite(du).all()) and all(p.grad is None for p in wrapper.parameters()))
            check(method+'_complete_WM_weights_norm_and_host_counts_unchanged',_tensor_digest(wrapper.wm.state_dict())==before_digest==receipt['WM_state_sha256'] and all(torch.equal(value,wrapper.wm.state_dict()[key]) for key,value in before.items()) and host_counts==(wrapper.wm.norm._host_count,wrapper.wm.delta_norm._host_count))
            check(method+'_inputs_and_RNG_unchanged',torch.equal(x,x_before) and torch.equal(u.detach(),u_before) and rng_equal(rng_before))
            check(method+'_checkpoint_context_files_unchanged',pin(path)==bytes_before and path.stat().st_mtime_ns==stat_before.st_mtime_ns and pin(run/'fixed_physics_context.npz')==context_before)
            np.savez_compressed(folder/'synthetic_forward_and_du.npz',x58=x.numpy(),u20=u.detach().numpy(),xi46=wrapper.physical_xi.numpy(),dynamics278=captured[2].numpy(),predicted_state58=x1.detach().numpy(),predicted_native_selfcollision_logits15=contact.detach().numpy(),du20=du.numpy())
            gradient_info={'du_l2':float(du.norm()),'du_maxabs':float(du.abs().max()),'nonzero_components':int((du!=0).sum()),'objective':'weighted sum of synthetic x1 (weights .1..1) + .01 * native selfcollision logits sum','physical_gradient_qualification':False}
            (folder/'load_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');methods[method]={'receipt':pin(folder/'load_receipt.json'),'checkpoint':bytes_before,'gradient_diagnostic':gradient_info,'forward_calls':1,'du_calls':1,'input_scope':'Synthetic validrotation state with frozenq0 and zeroqd/p/v/w; not a raw observed-state accuracy test'}
            if method=='A1L':
                # Invalid forward inputs fail before native model execution.
                rejected('reject_float64_control',lambda:wrapper(x,u.detach().double()))
                rejected('reject_wrong_state59',lambda:wrapper(torch.zeros(1,59),u))
                rejected('reject_wrong_control19',lambda:wrapper(x,torch.zeros(1,19)))
                rejected('reject_empty_batch',lambda:wrapper(torch.zeros(0,58),torch.zeros(0,20)))
                rejected('reject_nonfinite_input',lambda:wrapper(x,torch.full((1,20),float('nan'))))
                wrapper.train();rejected('reject_training_mode_normalizers',lambda:wrapper(x,u));wrapper.eval()
                check('invalid_calls_no_weights_norm_or_RNG_mutation',_tensor_digest(wrapper.wm.state_dict())==before_digest and host_counts==(wrapper.wm.norm._host_count,wrapper.wm.delta_norm._host_count))
        except BaseException as exc:errors.append({'method':method,'error':repr(exc),'traceback':traceback.format_exc()});check(method+'_actual_load_forward_completed',False,errors[-1])
    try:
        path=actual_checkpoints['A1L'];run=path.parents[1];health=ROOT/'outputs/health_primary_a1l_final_v1/report.json';state=torch.load(path,map_location='cpu',weights_only=False);minimal={'format_version':104,'contact_dim':15,'config':copy.deepcopy(state['config']),'modules':{'wm':dict(state['modules']['wm'])}}
        def payload_case(name,edit):
            value={'format_version':minimal['format_version'],'contact_dim':minimal['contact_dim'],'config':copy.deepcopy(minimal['config']),'modules':{'wm':dict(minimal['modules']['wm'])}};edit(value);rejected(name,lambda:_wm_payload(value))
        payload_case('reject_payload_old3',lambda s:s.update(format_version=3))
        payload_case('reject_payload_old103',lambda s:s.update(format_version=103))
        payload_case('reject_wrong_hidden',lambda s:s['config'].update(wm_hidden=[256,256,256]))
        payload_case('reject_taskcontext368_as_WM_input',lambda s:s['modules']['wm'].__setitem__('norm.mean',torch.zeros(388)))
        payload_case('reject_delta51',lambda s:s['modules']['wm'].__setitem__('delta_norm.mean',torch.zeros(51)))
        payload_case('reject_wrong_native_contact_dim',lambda s:s.update(contact_dim=14))
        payload_case('reject_nonfinite_WM_payload',lambda s:s['modules']['wm'].__setitem__('error_floor',torch.full((52,),float('nan'))))
        payload_case('reject_negative_normalizer_variance',lambda s:s['modules']['wm'].__setitem__('norm.var',torch.full((298,),-1.)))
        payload_case('reject_wrong_native_count_dtype',lambda s:s['modules']['wm'].__setitem__('norm.count',torch.tensor(0,dtype=torch.int64)))
        ctx=loadnp(run/'fixed_physics_context.npz')
        def context_case(name,edit):
            value={k:v.copy() for k,v in ctx.items()};edit(value);rejected(name,lambda:_physical_xi(value,b['context'],128))
        context_case('reject_xi368',lambda c:c.__setitem__('reader_context',np.zeros((128,368),np.float32)))
        context_case('reject_xi_float64',lambda c:c.__setitem__('reader_context',c['reader_context'].astype(np.float64)))
        context_case('reject_one_world_xi_ULP_drift',lambda c:c['reader_context'].__setitem__((1,0),np.nextafter(c['reader_context'][1,0],np.float32(np.inf))))
        context_case('reject_actual8_mass_drift',lambda c:c['body_mass'].__setitem__((0,0),c['body_mass'][0,0]+np.float32(1.)))
        context_case('reject_unrecognized_actual8_storage_rows',lambda c:c.__setitem__('body_mass',np.broadcast_to(c['body_mass'][:1],(2,)+c['body_mass'].shape[1:]).copy()))
        rejected('reject_model100_instead_of_final',lambda:load_world_model_checkpoint(run/'train/model_100.pt',b,'A1L',run,health))
        rejected('reject_wrong_training_run',lambda:load_world_model_checkpoint(path,b,'A1L',actual_checkpoints['P_ONLY'].parents[1],health))
        rejected('reject_wrong_expected_method',lambda:load_world_model_checkpoint(path,b,'P_ONLY',run,health))
        rejected('reject_prefix_health_for_final',lambda:load_world_model_checkpoint(path,b,'A1L',run,ROOT/'outputs/health_primary_a1l_100_v1/report.json'))
        bad=read(health);bad['total']+=1;bad_path=out/'bad_health_count_fixture.json';bad_path.write_text(json.dumps(bad));rejected('reject_inconsistent_health_counts',lambda:load_world_model_checkpoint(path,b,'A1L',run,bad_path))
    except BaseException as exc:errors.append({'section':'fixtures','error':repr(exc),'traceback':traceback.format_exc()});check('negative_fixture_section_completed',False,errors[-1])
    check('actual_three_final_methods_loaded_once',set(methods)=={'A1L','P_ONLY','A2'} and all(v['forward_calls']==v['du_calls']==1 for v in methods.values()))
    check('CPU_no_Warp_Env_GPU',not torch.cuda.is_initialized() and not any(name in sys.modules for name in ('warp','mujoco_warp','mjlab')))
    report={'schema':'rough104_world_model_checkpoint_CPU_contract:1','status':'PASS' if all(c['passed'] for c in checks) and not errors else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'methods':methods,'source_pins':source_pins,'errors':errors,'actual_final_WM_loads':len(methods),'native_forward_per_method':1,'du_gradient_per_method':1,'native_Env_or_Runner_constructed':False,'physics_steps':0,'CUDA_initialized':torch.cuda.is_initialized(),'Warp_imported':'warp' in sys.modules,'scope':'FP32CPU frozen native WM consumer and synthetic-input autograd only. Not observed-state prediction accuracy, GPUAMP parity, physical contact/force prediction, or physical-Jacobian qualification.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'checks':report['check_count'],'passed':report['passed'],'failed':[c for c in checks if not c['passed']]},indent=2));raise SystemExit(report['status']!='PASS')


if __name__=='__main__':main()
