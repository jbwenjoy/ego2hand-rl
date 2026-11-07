"""Saved initial parity review: no Torch/checkpoint/nativeLog/model reload.

Version2 preserves version1 and binds captured native s52 to both original
iteration1 saved arrays already checked against own initial by health29. NumPy
sqrt is diagnostic only, never a bit-exact native sqrt replacement.

Optional scalar FP32 means are checked against conservative reduction roundoff,
not falsely called exact NumPy replicas of a Torch reduction kernel. The public
subtotal preserves its original FP32 state+mode rounding before F64 impulse.
"""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import sys,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UTILITY=ROOT/'scripts/audit_component_task_gradient_saved_cpu_v1.py';UTILITY_SHA='472c72117dde51f7edd7c28d34301cef88939ebf6f5a64cb9bc6849dab42d0c5'
MATH=ROOT/'src/full151_gradient/boundary_impulse_independent_math_v1.py';MATH_SHA='48eb17c899491f303d681c862fbaebb512f51f6eab5d10bc03118e01ecf8bef4'
CORE=ROOT/'src/full151_gradient/boundary_component_initial_parity105_v1.py';CORE_SHA='2b5019b9bba1fe70258101dd35e77b7a1a691883ae53eeacd5fb9b9204af66e2'
PREREG=ROOT/'protocols/full151105_boundary_component_initial_parity_CPU_prereg_v1.json';PREREG_SHA='499ec25b5a9ad97c730a3b10ccbc94ecbbc624c2d324290bc14cf7b2632f5bdc'
READY=ROOT/'outputs/full151105_boundary_component_initial_parity_preparation_CPU_v1/READY.json';READY_SHA='bdb361f981f54faba674ed8a54e76d0fbaaa0657ad839ecfafbd769405c15cf2'
ARMS=('shared_public_losses','shared_public_plus_cross_target_consistency');ROLES=('unwrapped_baseline',)+ARMS


def module(p,h):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Frozen code changed '+str(p))
    s=importlib.util.spec_from_file_location('parity_saved_'+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


U=module(UTILITY,UTILITY_SHA)
pin=U.pin;sha=U.sha;read=U.read;checked=U.checked;same=U.same;load=U.load;bits=U.bits;dump=U.dump;require=U.require


def close(a,b):return np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,rtol=2e-12,atol=2e-14,equal_nan=True)
def positive_zero(a):return a.dtype==np.float32 and np.isfinite(a).all() and not a.view(np.uint32).any()


def tensor_digest_arrays(values):
    h=hashlib.sha256();names={np.dtype('f4'):'torch.float32',np.dtype('f8'):'torch.float64',np.dtype('i8'):'torch.int64'}
    for k,a in sorted(values.items()):
        require(a.dtype in names,'Stored native buffer dtype');h.update(k.encode());h.update(str((names[a.dtype],tuple(a.shape))).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def public_order(state,mode,impulse,consistency,coefficient):
    return float(np.float64(np.float32(np.float32(state)+np.float32(mode)))+np.float64(impulse)),float(np.float64(np.float32(np.float32(state)+np.float32(mode)))+np.float64(impulse)+coefficient*np.float64(consistency))


def fp32_mean_evidence(captured,squared32):
    a=np.asarray(squared32);require(a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all(),'Finite FP32 squared reduction operands')
    reference=math.fsum(float(x) for x in a.flat)/a.size;u=2.**-24;n=a.size+3;gamma=n*u/(1-n*u)
    # Conservative sequential-rounding envelope, including underflow allowance.
    bound=gamma*abs(reference)+n*float(np.nextafter(np.float32(0),np.float32(1)))
    return {'captured_FP32_mean':float(captured),'F64_mean_of_FP32_elements':reference,'difference':float(captured)-reference,'conservative_reduction_roundoff_bound':bound,'within_roundoff':abs(float(captured)-reference)<=bound,'bitexact_kernel_replay':False}


def captured_effective52(captured,prior_a,prior_b):
    for v in (captured,prior_a,prior_b):U.typed(v,(52,),'f4','captured original native scale52');require((v>0).all(),'Positive captured scale52')
    require(bits(captured,prior_a) and bits(captured,prior_b),'All52 current context bytes equal both prior initial-bound scale arrays')
    return captured.copy()


def metadata(protocol_path):
    require(sha(CORE)==CORE_SHA and sha(PREREG)==PREREG_SHA and sha(READY)==READY_SHA,'Current frozen runner/prereg/preparation')
    p,pr=read(protocol_path),read(PREREG);require(p['schema']=='full151105_boundary_component_initial_parity_CPU_protocol:1' and p['role']==pr['role'] and p['accuracy_gate'] is None,'Parity namespace/scope')
    require(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False and p['evaluate_auxiliary_losses'] is pr['evaluate_auxiliary_losses'],'CPU/optional diagnostic scope')
    require(same(p['preregistration'],pin(PREREG)) and set(p['inputs'])==set(pr['inputs']) and len(p['inputs'])==10,'Exact10 aliases')
    for k,v in p['inputs'].items():require(same(v,pr['inputs'][k]),'Same fixed input '+k)
    for v in p['source_pins'].values():checked(v)
    require(any(same(v,pin(CORE)) for v in p['source_pins'].values()),'Current runner execution source')
    i=p['inputs'];target=read(checked(i['target_report']));review=read(checked(i['target_saved8']));U.allchecks(review)
    require(target['enrollment_qualified'] is True and all(v is True for v in target['guards'].values()) and review['engineering_qualified'] is True and same(review['producer_report'],i['target_report']),'Actual all152 conditional dataset source')
    health=read(checked(i['initial_saved_health29']));U.allchecks(health);require(same(health['initial_component'],i['initial_component']),'Original own initial role')
    ready=read(READY);proof=read(checked(ready['cpu_report']));U.allchecks(proof);require(proof['total']==3 and ready['source_pins']==proof['source_pins'],'Focused final runner toy3')
    for v in ready['source_pins'].values():checked(v)
    return p


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_parity_read.json',{'prior_failed_v1_preserved':True,'scale_basis':'captured native s52 exact to prior serialized-initial-bound evidence; NumPy sqrt diagnostic only','auditor':auditor,'aux_math':pin(MATH),'actual_parity_arrays_read':False,'F32_public_order':'F32(state+mode) then F64 impulse and consistency','FP32_reduction_check':'F64 element reference and gamma_(N+3), unit roundoff2^-24; arithmetic envelope only, no scientific accuracy gate','F64_compare_rtol':2e-12,'F64_compare_atol':2e-14})
    p=metadata(protocol_path);i=p['inputs'];r=read(run/'report.json');sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json');checks={};details={};m=module(MATH,MATH_SHA)
    def gate(k,v):checks[k]=bool(v)
    for v in [*sources.values(),*outputs.values()]:checked(v)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('authorized_protocol_all10aliases_original_sources_ZIP_outputpins',p['CPU_run_authorized'] is True and same(r['protocol'],pin(protocol_path)) and p==read(run/'protocol.json') and r['input_pins']==i and zipok)
    data=load(run/'prepared_inputs_and_context.npz');source=load(checked(i['target_arrays']));inputs=load(run/'actual_forward_inputs152.npz');ctx=load(run/'initial_frozen_context_buffers.npz');norm=load(run/'initial_normalizer_buffers.npz');receipt=read(checked(i['initial_validation']))
    shapes={'x58_FP32':(152,58),'u20_FP32':(152,20),'target58_FP32':(152,58),'physical_xi46_FP32':(46,),'palm_rot_raw_FP32':(152,3,3),'gravity_palm_F64':(152,3),'DeltaT_F64':(152,),'mass_actual_F64':(),'I_body_F64':(3,3),'J_palm_F64':(152,2,3),'K_palm_F64':(152,2,3),'r_v_saved_m_s_F64':(152,3),'r_w_saved_rad_s_F64':(152,3),'boundary_constraint_witness':(152,2)}
    require(set(data)==set(shapes),'Exact prepared operand names')
    for k,shape in shapes.items():U.typed(data[k],shape,'?' if k=='boundary_constraint_witness' else 'f4' if k.endswith('FP32') else 'f8',k)
    require(all(bits(v,source[k]) for k,v in data.items()),'All prepared dataset source bytes')
    x,u,target=data['x58_FP32'],data['u20_FP32'],data['target58_FP32'];xi=data['physical_xi46_FP32']
    expected_dyn=np.concatenate((x,np.broadcast_to(xi,(152,46)),np.zeros((152,174),dtype='f4')),axis=-1)
    require(set(inputs)=={'x58_FP32','actual_u20_FP32','actual_dynamics278_FP32'},'No targets/labels in actual forward signature')
    gate('all152_actual_forward_state_u_xi_zero174_and_input_digest',bits(inputs['x58_FP32'],x) and bits(inputs['actual_u20_FP32'],u) and bits(inputs['actual_dynamics278_FP32'],expected_dyn) and tensor_digest_arrays({'x':x,'u':u,'dynamics':expected_dyn,'target_outside_forward':target})==r['input_digest_before']==r['input_digest_after'])
    hosts=r['normalizer_host_configuration_before'];eps=hosts['unwrapped_baseline']['delta_norm']['eps']
    numpy_std_diagnostic=np.maximum(np.sqrt(norm['delta_norm.var']+np.float32(eps**2)).astype('f4'),norm['error_floor'])
    old_run=checked(i['initial_component']).parent;old_health=read(checked(i['initial_saved_health29']));old_pins=read(old_run/'output_pins.json')
    require(Path(old_health['run_dir']).samefile(old_run) and all(old_health['checks'][k] is True for k in ('frozen_s52_recomputed_native_FP32_std_floor','pointwise_iteration1_PREoptimizer_loss_arithmetic','pointwise_plus_common_pre_response_iteration1_PREoptimizer_loss_arithmetic')),'Previous serialized initial and loss-evidence native scale checks')
    prior_scales=[];scale_sources={}
    for arm in ('pointwise','pointwise_plus_common_pre_response'):
        filename=arm+'_iteration_0001_preupdate_loss_evidence.npz';q=old_pins[filename]
        require(q['sha256']=='b51f83d56db7b5358eaccef062fb0f5b1972fb98bcc58ab58ee43b34c79e2f2b' and checked(q).samefile(old_run/filename),'Frozen prior native scale evidence route/SHA')
        prior_scales.append(load(checked(q))['scale52_FP32']);scale_sources[arm]=q
    effective=captured_effective52(ctx['effective_s52_FP32'],*prior_scales)
    scale_difference=numpy_std_diagnostic.astype('f8')-effective.astype('f8');scale_differing_indices=np.flatnonzero(numpy_std_diagnostic.view('u4')!=effective.view('u4'))
    details['scale52_evidence']={'source_pins':scale_sources,'previous_serialized_initial_health':i['initial_saved_health29'],'primary':'All52 captured native context bytes exact to both original iteration1 evidence arrays; prior health29 binds them to own initial native sqrt/floor. Reviewer does not replay Torch sqrt.','NumPy_sqrt_diagnostic_only':{'values_FP32_as_F64':numpy_std_diagnostic.tolist(),'captured_native_FP32_as_F64':effective.tolist(),'difference_F64':scale_difference.tolist(),'different_component_indices':scale_differing_indices.tolist(),'signed_ULP_difference':[int(x)-int(y) for x,y in zip(numpy_std_diagnostic.view('u4'),effective.view('u4'))],'is_qualification_gate':False}}

    fixed={'mass_actual_FP32':np.asarray(data['mass_actual_F64'],dtype='f4'),'I_body_F64':data['I_body_F64'],'gravity_palm_F64':data['gravity_palm_F64'][0],'DeltaT_F64':data['DeltaT_F64'][0],
      'palm_rot_FP32':data['palm_rot_raw_FP32'][0],'physical_xi46_FP32':xi,'effective_s52_FP32':effective,'I_body_antisymmetry_maxabs_F64':np.asarray(abs(data['I_body_F64']-data['I_body_F64'].T).max())}
    allfixed=all(bits(data[k],np.broadcast_to(data[k][:1],data[k].shape).copy()) for k in ('gravity_palm_F64','DeltaT_F64','palm_rot_raw_FP32'))
    gate('all152_fixed_context_effective52_raw_norm_and_initial_receipt',allfixed and set(ctx)==set(fixed) and all(bits(v,ctx[k]) for k,v in fixed.items()) and tensor_digest_arrays(norm)==receipt['frozen_norm_sha256'] and r['initial_WM_sha256']==receipt['initial_WM_sha256'])
    f={role:load(run/(role+'_forward152.npz')) for role in ROLES};base=f[ROLES[0]];masks=load(run/'all152_parity_masks.npz');allparity=True
    for role in ROLES:
        for key,shape in [('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15))]:U.typed(f[role][key],shape,'f4',role+'/'+key)
    for arm in ARMS:
        o=f[arm]
        for key in ('prediction58_FP32','self15_logits_FP32'):
            expected=(o[key].view(np.uint32)==base[key].view(np.uint32)).all(axis=1);saved=masks[arm+'__'+key+'_bit_equal'];allparity &= bits(expected,saved) and expected.all()
        for key,shape in [('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))]:
            U.typed(o[key],shape,'f4',key);expected=(o[key].view(np.uint32).reshape(152,-1)==0).all(axis=1);allparity &= bits(expected,masks[arm+'__'+key+'_exact_positive_zero']) and positive_zero(o[key])
    gate('all3x152_native58_self15_exactbits_and_all14_positivezero_masks',allparity)
    storage=read(run/'named_model_context_storage_inventory.json');require(set(storage)==set(ROLES),'All3 simultaneous model inventories')
    sets=[];storageok=True
    for role,rows in storage.items():
        require(rows and len({(v['kind'],v['name']) for v in rows})==len(rows),'Named storage records unique')
        one=set()
        for v in rows:
            require(v['device']=='cpu' and v['kind'] in ('parameter','buffer') and all(type(v[k]) is int and v[k]>=0 for k in ('storage_base_ptr','data_ptr','storage_nbytes','storage_offset')),'Observed CPU named storage ABI')
            size={'torch.float32':4,'torch.float64':8}[v['dtype']];storageok &= v['data_ptr']==v['storage_base_ptr']+v['storage_offset']*size
            if v['storage_nbytes']>0:one.add(v['storage_base_ptr'])
        sets.append(one)
    storageok &= all(not(sets[a]&sets[b]) for a in range(3) for b in range(a+1,3))
    construction=r['construction_receipt'];expected_count={'forward_attempted':1,'native_forward_returned':1,'hook_calls':1,'head_forward_returned':1,'forward_returned':1}
    gate('named_storage_bases_disjoint_and_each_hook_native_head_once_cache_scope',storageok and all(r['wrapper_counts'][arm]==expected_count for arm in ARMS) and construction['raw_WM_unchanged'] is True and all(construction['raw_clone_state_identity'].values()) and len(set(construction['zero_head_digests'].values()))==1 and construction['feature_hooks_independent'] is True)
    rng=U.raw_rng(run);gate('raw_Python_NumPy_TorchCPU_RNG_headers_signedfloat_and_bytes_equal',all(rng.values()) and all(v is True for v in r['raw_RNG_value_and_byte_equality'].values()))
    loss_checks=True;loss_evidence={}
    if p['evaluate_auxiliary_losses']:
        sv=effective[46:49].astype('f8');sw=effective[49:52].astype('f8');Ib=data['I_body_F64'];mass=float(ctx['mass_actual_FP32']);g=data['gravity_palm_F64'];dt=data['DeltaT_F64']
        for ai,arm in enumerate(ARMS):
            e=load(run/(arm+'_descriptive_losses.npz'));pred=f[arm]['prediction58_FP32'];z=f[arm]['z12_FP32'];D=e['native_point_delta52_FP32'];U.typed(D,(152,52),'f4','captured native difference')
            for lo,hi,slo,shi in [(0,40,0,40),(40,43,40,43),(46,49,52,55),(49,52,55,58)]:loss_checks &= bits(D[:,lo:hi],target[:,slo:shi]-pred[:,slo:shi])
            chart=np.linalg.norm(D[:,43:46].astype('f8'),axis=1);loss_checks &= np.isfinite(chart).all() and (chart<np.pi-.01).all()
            squares=np.square(np.divide(D,effective,dtype='f4'),dtype='f4');point=fp32_mean_evidence(e['L_state_FP32'],squares)
            mode=fp32_mean_evidence(e['L_mode_FP32'],np.full((152,2),np.float32(math.log(2)),dtype='f4'))
            targets=[];Jhat=[];Khat=[];rv=[];rw=[];I0s=[];I1s=[];rhsall=[];solall=[]
            for j in range(152):
                I0=m.congruence_native_R(x[j,43:52].astype('f8').reshape(3,3),Ib);I1=m.congruence_native_R(pred[j,43:52].astype('f8').reshape(3,3),Ib)
                tz=np.concatenate(((data['J_palm_F64'][j]/mass)/sv,np.asarray([m.solve3_partial_pivot(I0,v)/sw for v in data['K_palm_F64'][j]])),axis=1)
                J=mass*(sv*z[j,:,:3].astype('f8'));K=np.asarray([m.matvec(I0,sw*v) for v in z[j,:,3:].astype('f8')]);rhs=m.matvec(I0,x[j,55:58].astype('f8'))+K.sum(0);sol=m.solve3_partial_pivot(I1,rhs)
                targets.append(tz);Jhat.append(J);Khat.append(K);rv.append(pred[j,52:55].astype('f8')-x[j,52:55].astype('f8')-(J.sum(0)+mass*g[j]*dt[j])/mass);rw.append(pred[j,55:58].astype('f8')-sol);I0s.append(I0);I1s.append(I1);rhsall.append(rhs);solall.append(sol)
            tz,J,K,rv,rw,I0,I1,rhs,sol=map(np.asarray,(targets,Jhat,Khat,rv,rw,I0s,I1s,rhsall,solall));imp=float(np.mean((z.astype('f8')-tz)**2));cons=float(np.mean(np.concatenate(((rv-data['r_v_saved_m_s_F64'])/sv,(rw-data['r_w_saved_rad_s_F64'])/sw),axis=1)**2))
            public,total=public_order(e['L_state_FP32'],e['L_mode_FP32'],imp,cons,float(ai))
            expected={'J_hat_palm_F64':J,'K_hat_palm_F64':K,'z_target_F64':tz,'r_v_hat_F64':rv,'r_w_hat_F64':rw,'I_palm0_F64':I0,'I_palm1_hat_F64':I1,'angular_RHS_F64':rhs,'omega_from_predicted_impulses_F64':sol,'L_imp_F64':np.asarray(imp),'L_cons_F64':np.asarray(cons),'L_public':np.asarray(public),'L_total':np.asarray(total)}
            expected['solve_residual_F64']=np.asarray([m.matvec(A,v)-b for A,v,b in zip(I1,sol,rhs)])
            loss_checks &= point['within_roundoff'] and mode['within_roundoff'] and all(close(e[k],v) for k,v in expected.items())
            ds=r['descriptive_initial_losses'][arm];loss_checks &= float(e['L_state_FP32'])==ds['native_point_FP32'] and float(e['L_mode_FP32'])==ds['mode_BCE_FP32'] and close(imp,ds['impulse_z_MSE_F64']) and close(cons,ds['saved_defect_consistency_MSE_F64']) and close(public,ds['public_loss']) and close(total,ds['diagnostic_total'])
            loss_evidence[arm]={'point_reduction':point,'mode_reduction':mode,'L_imp_F64':imp,'L_cons_F64':cons,'L_public_source_order':public,'L_total':total,'native_rotvec_norm_max_from_capture':float(chart.max()),'rotational_Log_source_capture_not_recomputed':True}
    gate('optional_saved_native52_arithmetic_F64_aux_losses_original_F32_subtotal',loss_checks)
    expected_counts={'trusted_initial_torch_load_attempted':1,'trusted_initial_torch_load_returned':1,'native_WM_constructors':1,'raw_deepcopies':2,'WM_forward_attempted':3,'WM_forward_returned':3,'predicted_rows':456,'AD_optimizer_train_checkpoint_old119f_Actor_MJB_Env_physics_GPU':0}
    gate('source_weights_norm_host_caches_and_complete_fixed_execution_budget',r['counts']==expected_counts and r['state_digests_before']==r['state_digests_after'] and hosts==r['normalizer_host_configuration_after'] and all(v==receipt['normalizer_host_configuration'] for v in hosts.values()) and all(v is True for v in r['invariants'].values()))
    gate('engineering_function_identity_only_no_training_native105_or_physical_upgrade',r['schema']=='full151105_boundary_component_initial_parity_CPU_analysis:1' and r['status']=='COMPLETED_REAL_INITIAL_FUNCTION_PARITY_NOT_TRAINING_QUALIFICATION' and same(r['initial_component'],i['initial_component']) and r['function_identity_qualified'] is True and all(r[k] is False for k in ('native105_or_training_health_qualified','actual_training_authorized','new_loss_recipe_frozen','physical_force_or_gradient_qualified','whole_hidden_state_qualified')))
    for v in [*sources.values(),*i.values(),*outputs.values(),*scale_sources.values()]:checked(v)
    gate('reviewer_no_Torch_checkpoint_nativeLog_models_FW_AD_or_physics',not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor)
    details.update(raw_RNG=rng,optional_loss_arithmetic=loss_evidence,source_files=len(sources),weight_scope='State/parameter/normalizer/host invariance remains source-backed digest equality, not a reviewer full state reload. Raw initial normalizer bytes independently bound to original receipt.',storage_scope='Simultaneous named parameter/buffer underlying storage-base metadata relation; no hidden/opaque-state certificate.',F32_mean_scope='Conservative sequential gamma_(N+3) envelope from saved FP32 element operands; not bit-exact Torch kernel replay or a scientific accuracy threshold.')
    result={'schema':'full151105_boundary_initial_parity_saved_independent:1','status':'PASS_SAVED_REAL_INITIAL_FUNCTION_PARITY' if all(checks.values()) else 'FAILED_SAVED_INITIAL_PARITY_REVIEW','engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_auditor':auditor,'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'initial_component':i['initial_component'],'function_identity_qualified':all(checks.values()),'counts':r['counts'],'details':details,'actual_training_authorized':False,'new_loss_recipe_frozen':False,'whole_hidden_state_qualified':False,'physical_force_or_gradient_qualified':False,'reviewer_Torch_checkpoint_loads_nativeLog_models_FW_AD_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();metadata(a.protocol)
    if not a.run:print(json.dumps({'status':'PASS_METADATA_ONLY_NO_REAL_PARITY_ARRAYS','auditor':pin(__file__)}));return 0
    require(a.run_dir is not None and a.output is not None,'Fresh actual run/output required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.output.resolve())
    except Exception as e:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'producer_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
