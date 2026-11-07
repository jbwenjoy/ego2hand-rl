"""Prospective saved-only precision audit. No model/Env/native/GPU import."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import pickle
import sys
import time
import zipfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_A2_GPU_precision_v1'
WATCH=RUN.with_name(RUN.name+'_watchdog')
OUT=ROOT/'outputs/full151105_GPU_precision_saved_independent_v1'
MODES=('FP32_STORAGE_TF32_ALLOWED','BF16_NET_FP32_INTEGRATE_TF32_ALLOWED')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def matches(v):return pin(v['path'])['sha256']==v['sha256'] and os.path.samefile(Path(v['path']).resolve(),v['path'])
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b):return a is b if a is None or b is None else bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))
def diff(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    return dict(maximum_absolute=float(abs(d).max()),mean_absolute=float(abs(d).mean()),RMS=float(np.sqrt(np.mean(d*d))))
def metrics(pred,truth):
    d=pred.astype(float)-truth.astype(float);R=pred[:,43:52].astype(float).reshape(-1,3,3);T=truth[:,43:52].astype(float).reshape(-1,3,3);angle=np.arccos(np.clip((np.einsum('bij,bij->b',R,T)-1)*.5,-1,1));rn=lambda v:float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
    return dict(q_component_RMSE_mrad=float(np.sqrt(np.mean(d[:,:20]**2))*1000),qd_component_RMSE_rad_s=float(np.sqrt(np.mean(d[:,20:40]**2))),p_vector_RMS_mm=rn(d[:,40:43])*1000,R_geodesic_RMS_degrees=float(np.rad2deg(np.sqrt(np.mean(angle**2)))),v_vector_RMS_m_s=rn(d[:,52:55]),omega_vector_RMS_rad_s=rn(d[:,55:58]))
def terms64(x,p,R):
    x,p,R=np.asarray(x,float),np.asarray(p,float),np.asarray(R,float);a=x[40:43];r=x[43:52].reshape(3,3);e=.5*(np.linalg.norm(a+.1*r[:,0]-p-.1*R[:,0])+np.linalg.norm(a-.1*r[:,0]-p+.1*R[:,0]))
    return np.array([2*np.exp(-(np.linalg.norm(a-p)/.05)**2),8*np.exp(-(e/.04)**2)])
class StateOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray,('numpy._core.numeric','_frombuffer'):np._core.numeric._frombuffer,('numpy.core.numeric','_frombuffer'):np._core.numeric._frombuffer,('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct}
        if (module,name) not in allowed:raise ValueError('Host RNG pickle contains an unexpected global: '+module+'.'+name)
        return allowed[(module,name)]

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();checks={};detail={}
    def check(k,v):checks[k]=bool(v)
    watch=read(WATCH/'state.json');rootconfig=pin(watch['command'][4]);c=read(rootconfig['path']);r=read(RUN/'report.json');rt=read(RUN/'runtime.json');ident=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json');inv=read(RUN/'source/inventory.json')
    check('rootconfig_f5e20_and_saved_copies_exact',rootconfig['sha256']=='f5e20fe21cae8e62d507533f81fb6c7704da0f18fa82717214db952c01febe50' and pin(RUN/'config.json')['sha256']==pin(WATCH/'config_snapshot.json')['sha256']==rt['plan_sha256']==rootconfig['sha256'] and r['config']==rootconfig)
    check('source_closure_all_current_and_config_exact',inv==c['source_pins'] and all(matches(v) for v in inv.values()))
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        names=[f'{i:04d}_{Path(v["path"]).name}' for i,v in enumerate(inv.values())]
        check('execution_ZIP_all_members_bytes_exact',z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in zip(names,inv.values())))
    check('module_b292_and_actual_entry_source_bound',any(v['sha256']=='b292630b290dd2d05f2987c3f0156bb2d301968680e8282ae086f66fb498964f' for v in inv.values()) and pin(watch['command'][2])['sha256']==rt['source_sha256'] and any(v['sha256']==rt['source_sha256'] for v in inv.values()))
    check('completed_worker_watchdog_OS12_no_timeout',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and rt['status']=='completed_full151_GPU_precision' and ident['engineering_identity_qualified'] is True and len(ident['checks'])==12 and all(v is True for v in ident['checks'].values()) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    obs=watch['actual_process_OS_observation'];cmd=watch['command'];py=ident['selected_python_identity'];check('raw_PID_parent_executable_command_rebound',obs['pid']==rt['pid']==watch['actual_pid']>0 and obs['parent_pid']==rt['parent_pid']==watch['launcher_pid']>0 and obs['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:]) and os.path.samefile(obs['executable_path'],py['base_executable_raw']) and Path(cmd[6]).resolve()==RUN.resolve())
    check('report_result_completion_scope_matches',r['schema']=='full151105_GPU_precision_diagnostic_report:1' and r['status']=='COMPLETED_GPU_PRECISION_DIAGNOSTIC' and read(RUN/'result.json')==r and read(RUN/'completion.json')['invariants']==r['invariants'] and not (RUN/'failure.json').exists())
    proto=read(c['protocol']['path']);review=read(proto['source_review']['path']);check('prereg3a217_source21_and_flags_scope',matches(c['protocol']) and c['protocol']['sha256']=='3a217f419784a71e9350589126fd738fc3d6bce2876c0316aed874abaf4ce5bc' and proto['schema']=='full151105_GPU_precision_preregistration:1' and proto['execution_authorized'] is False and proto['modes']==list(MODES) and proto['allow_tf32'] is True and review['passed']==review['total']==21 and all(v is True for v in review['checks'].values()) and all(matches(v) for v in proto['source_pins'].values()))
    check('fixed_modes_and_2forward8_2AD_2read_1strictload_counts',r['modes']==c['modes']==list(MODES) and all(type(r[k]) is int and r[k]==n for k,n in {'strict_CPU_WM_loader_calls':1,'actual_checkpoint_reads':2,'WM_forward_calls':2,'predicted_rows':16,'u_autograd_calls':2,'physical_comparison_rows_per_mode':64,'dependent_direction_groups':32,'physics_steps':0,'optimizer_updates':0,'Actor_forwards':0}.items()))
    check('scope_GPU_diagnostic_not_Env_or_physical_certificate',r['GPU_used'] is True and r['Env_created'] is False and r['compile_used'] is False and r['physical_gradient_qualified'] is False and r['whole_hidden_state_qualified'] is False and r['historical_training_runtime_flags_observed'] is False and r['source_intended_training_allow_tf32'] is True)
    invariants=r['invariants'];check('all_runtime_invariance_assertions_true',all(type(v) is bool and v for v in invariants.values()) and set(invariants)=={'weights_and_normalizers_unchanged','normalizers_unchanged','CPU_RNG_unchanged','GPU_RNG_unchanged','inputs_unchanged','flags_held_between_modes','normalized_features_bitexact','flags_held_within_modes','parameters_have_no_grad','no_Warp'})
    for key in ('CPU_prerequisite_config','CPU_report','CPU_saved_review','CPU_output_pins','CPU_predictions','protocol'):check(key+'_pin_current',matches(c[key]))
    cpuconfig=read(c['CPU_prerequisite_config']['path']);cpureport=read(c['CPU_report']['path']);cpureview=read(c['CPU_saved_review']['path']);old=load(c['CPU_predictions']['path']);a=load(cpuconfig['artifact_pins']['physical_arrays']['path']);base=read(Path(c['CPU_report']['path']).parent/'comparisons.json');classes=read(Path(c['CPU_report']['path']).parent/'classifications.json')
    check('actual_CPU656_review_and_same_arrays_source_join',cpureview['schema']=='full151105_WM_analysis_saved_independent:1' and cpureview['passed']==cpureview['total']==656 and all(v is True for v in cpureview['checks'].values()) and cpureview['config']==c['CPU_prerequisite_config'] and cpureview['analysis_report']==c['CPU_report'] and cpureview['analysis_arrays']==c['CPU_predictions'] and len(base)==64 and len(classes)==32 and bits(old['snapshot_x'],a['snapshot_x']) and bits(old['actualu'],a['snapshot_u']))
    check('all_CPU_outputpins_unchanged',all(matches(v) for v in read(c['CPU_output_pins']['path']).values()))
    loader=read(RUN/'strict_CPU_loader_receipt.json');oldloader=read(Path(c['CPU_report']['path']).parent/'loader_receipt.json')
    check('strict119f_loader_same_finalweights_context_and_no_GPU_input_wrapper',loader['schema']=='full151_105_world_model_CPU_evaluation_load:1' and loader['status']=='PASS_FROZEN_CPU_WORLD_MODEL_LOAD' and loader['checkpoint']==cpuconfig['checkpoint'] and matches(loader['checkpoint']) and loader['WM_state_sha256']==oldloader['WM_state_sha256']==r['weights_before']==r['weights_after'] and loader['physical_xi_dim']==46 and loader['disturbance_dim']==174 and loader['WM_total_input_dim']==298 and loader['method']=='A2' and loader['iteration']==2000 and loader['amp_enabled_for_CPU_diagnostic'] is False)
    check('normalizer_before_after_digest_assertion_bound',r['normalizers_before']==r['normalizers_after'] and all(type(r[k]) is str and len(r[k])==64 for k in ('weights_before','weights_after','normalizers_before','normalizers_after')))
    arrays=load(RUN/'precision_arrays.npz');suffixes={'predicted_x':(8,58),'self15_logits':(8,15),'gradient_u':(8,20),'predicted_score':(8,),'score_terms':(8,2),'normalized_features':(8,298)}
    check('all16predictions_gradients_features_F32finite_exactABI',set(arrays)=={m+'__'+s for m in MODES for s in suffixes} and all(arrays[m+'__'+s].dtype==np.float32 and arrays[m+'__'+s].shape==shape and np.isfinite(arrays[m+'__'+s]).all() for m in MODES for s,shape in suffixes.items()))
    check('normalized_features8x298_bitexact_between_modes',bits(arrays[MODES[0]+'__normalized_features'],arrays[MODES[1]+'__normalized_features']))
    before=r['actual_flags_per_mode'];after=r['actual_flags_after_modes'];flags={'matmul_allow_tf32','float32_matmul_precision','allow_bf16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction_split_k','allow_fp16_reduced_precision_reduction','allow_fp16_accumulation','deterministic_algorithms','deterministic_warn_only','cudnn_allow_tf32','cudnn_deterministic','cudnn_benchmark','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUDA_VISIBLE_DEVICES'}
    check('all_mode_flags_before_after_present_identical_sharedTF32',len(before)==len(after)==2 and all(set(v)==flags and v['matmul_allow_tf32'] is True for v in before+after) and all(exact(before[0],v) for v in before+after))
    check('actual_net_dtype_F32_then_BF16_FP32_integration_outputs',r['models'][MODES[0]]['native_net_output_dtype']=='torch.float32' and r['models'][MODES[1]]['native_net_output_dtype']=='torch.bfloat16' and r['models'][MODES[0]]['amp'] is False and r['models'][MODES[1]]['amp'] is True)
    rng=load(RUN/'RNG_arrays.npz');check('rawCPU_CUDA_RNG_arrays_samebytes_oneGPU',set(rng)=={'torch_CPU_before','torch_CPU_after','CUDA_before_0','CUDA_after_0'} and all(v.dtype==np.uint8 and v.ndim==1 and v.size>0 for v in rng.values()) and bits(rng['torch_CPU_before'],rng['torch_CPU_after']) and bits(rng['CUDA_before_0'],rng['CUDA_after_0']))
    host=StateOnlyUnpickler(io.BytesIO((RUN/'RNG_host_states.pkl').read_bytes())).load();check('raw_hostPython_and_NumPy_RNG_states_equal',set(host)=={'before','after'} and len(host['before'])==len(host['after'])==2 and host['before'][0]==host['after'][0] and host['before'][1][0]==host['after'][1][0]=='MT19937' and bits(host['before'][1][1],host['after'][1][1]) and host['before'][1][2:]==host['after'][1][2:])
    cpu_hashes=[hashlib.sha256(pickle.dumps(host[t],protocol=5)+rng['torch_CPU_'+t].tobytes()).hexdigest() for t in ('before','after')]
    check('raw_CPU_RNG_payloads_reproduce_saved_hashes',cpu_hashes[0]==r['CPU_RNG_before_sha256']==r['CPU_RNG_after_sha256']==cpu_hashes[1])
    summaries={};projections=[]
    for mi,mode in enumerate(MODES):
        pred=arrays[mode+'__predicted_x'];grad=arrays[mode+'__gradient_u'];score=arrays[mode+'__predicted_score'];terms=arrays[mode+'__score_terms'];rows=read(RUN/(mode+'_comparisons.json'));model=r['models'][mode]
        check(mode+'__64rows_same_physical_FD_classes_only_AD_replaced',len(rows)==64 and all(exact({k:v for k,v in row.items() if k not in ('WM_AD_effective','WM_AD_nominal')},{k:v for k,v in b.items() if k not in ('WM_AD_effective','WM_AD_nominal')}) for row,b in zip(rows,base)))
        check(mode+'__score_is_F32_two_term_sum',bits(terms.sum(-1,dtype=np.float32),score))
        arithmetic=[]
        for s in range(8):
            t=terms64(pred[s],a['goal_p'][s],a['goal_R'][s]);bound=8*np.finfo(np.float32).eps*np.maximum(1.,abs(t));arithmetic.append(float(abs(t-terms[s]).max()));check(mode+'__score'+str(s)+'_same20_goal_F64_diagnostic',np.all(abs(t-terms[s].astype(float))<=bound))
        for i,(row,b) in enumerate(zip(rows,base)):
            s=row['snapshot_id'];ji=(0,4,8,12).index(row['joint_index']);ei=row['epsilon_index'];eps=(.002,.001)[ei];plus=a['direction__plus'][s,ji,ei];minus=a['direction__minus'][s,ji,ei];d=(plus.astype(float)-minus.astype(float))/(2*eps);proj=float(grad[s].astype(float)@d);nominal=float(grad[s,row['joint_index']]);projections.append(abs(proj-row['WM_AD_effective']))
            rid=s*19+3+ei*8+ji*2;fd=(float(a['branch_object_score_k'][rid+1])-float(a['branch_object_score_k'][rid]))/(2*eps)
            check(mode+'__projection'+str(i)+'_F64_actual_endpoint_noEMA',row['WM_AD_effective']==proj and row['WM_AD_nominal']==nominal and row['physical_slope']==fd and exact(row['d_eff'],d.tolist()))
        met=metrics(pred,a['donor_next_x']);met['score_RMS']=diff(score,a['donor_object_score_k'])['RMS'];check(mode+'__prediction_metrics_units_independent',all(close(v,model['prediction_metrics'][k]) for k,v in met.items()))
        cpumet=metrics(pred,old['predicted_x']);check(mode+'__CPU_state_delta_units_and_gradient_statistics',all(close(v,model['CPU_state_difference_separate_units'][k]) for k,v in cpumet.items()) and all(close(v,model['CPU_gradient_difference'][k]) for k,v in diff(grad,old['gradient_u']).items()))
        agree={}
        for ei,eps in enumerate((.002,.001)):
            selected=[v for v in rows if v['epsilon_index']==ei and v['stable_nonflat']];x=np.array([v['physical_slope'] for v in selected]);y=np.array([v['WM_AD_effective'] for v in selected]);n=len(selected);g=dict(eligible=n,sign_correct=int((np.sign(x)==np.sign(y)).sum()),sign_fraction=None if n==0 else float((np.sign(x)==np.sign(y)).mean()),pearson=None if n<2 or np.ptp(x)==0 or np.ptp(y)==0 else float(np.corrcoef(x,y)[0,1]),mean_absolute_disagreement=None if n==0 else float(abs(x-y).mean()),clipped_eligible=sum(bool(v['clipped']) for v in selected));agree[str(eps)]=g;check(mode+'__agreement'+str(eps)+'_same_groups_no_selection',all(close(v,model['agreement'][str(eps)][k]) for k,v in g.items()))
        summaries[mode]={'prediction_metrics':met,'CPU_gradient_difference':diff(grad,old['gradient_u']),'agreement':agree,'scalar_F64_maxerror':max(arithmetic)}
    pair=diff(arrays[MODES[1]+'__gradient_u'],arrays[MODES[0]+'__gradient_u']);check('mode_pair_gradient_difference_statistics',all(close(v,r['mode_pair_gradient_difference'][k]) for k,v in pair.items()))
    pins=read(RUN/'output_pins.json');check('all_immutable_module_output_pins_exact',all(matches(v) and Path(v['path']).resolve().parent==RUN.resolve() for v in pins.values()) and 'runtime.json' not in pins and all(k in pins for k in ('report.json','precision_arrays.npz','RNG_arrays.npz','RNG_host_states.pkl','strict_CPU_loader_receipt.json')))
    check('no_reloaded_WM_Actor_native_Env_GPU_or_Torch',not any(n.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic','scipy') for n in sys.modules))
    detail.update(source_count=len(inv),modes=summaries,mode_pair_gradient_difference=pair,projection_maxerror=max(projections),actual_flags=before[0],rawRNG_independently_equal=True,source_runtime_weight_digest_assertions_only=True)
    result={'schema':'full151105_GPU_precision_saved_independent:1','status':'PASS_SAVED_NUMERICAL_MODE_COMPARISON_NO_PHYSICAL_GRADIENT_CERTIFICATE' if all(checks.values()) else 'SAVED_AUDIT_FAILURE_RETAINED','engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(Path(__file__)),'run_dir':str(RUN.resolve()),'config':rootconfig,'producer_report':pin(RUN/'report.json'),'details':detail,'inputs':{p.name:pin(p) for p in (RUN/'precision_arrays.npz',RUN/'RNG_arrays.npz',RUN/'RNG_host_states.pkl',RUN/'source/inventory.json',RUN/'source/execution_sources.zip',WATCH/'state.json',WATCH/'engineering_identity.json')},'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'HO_gradient_qualified':False,'limits':['Both modes share the same frozen weights/input centers/physical FD rows, but each mode has only one GPU pass. There is no within-mode variability estimate or unique attribution of all differences to arithmetic precision. CPU-vs-GPU also conflates backend/hardware/TF32.','FP32 mode means FP32 storage with TF32 allowed, not strict IEEE GEMM. BF16 names the native net autocast stage; integration/output/score/gradient buffers stay FP32. Actual recorded flags and net-output dtype are source-backed observations, not a proof of every internal hardware instruction.','Raw CPU/CUDA RNG arrays and restricted host RNG pickle states are independently compared. Weight/normalizer before/after digest equality and input/parameter-gradient checks remain runtime source-backed assertions; no weights/model is loaded by this reviewer.','Physical classes and all64rows per mode are retained unchanged, including clipped/offset cases. The same32groups and same8states are dependent reused observations, not128independent samples. No threshold, best-mode selection, method-effect or HO/human-label conclusion is introduced.','Frozen119f validates on CPU before GPU initialization and internally constructs a CPU Actor for signature validation; zero Actor forwards is correct, zero Actor constructors is not claimed. No new Env, Warp, physics, training, compile or model inference occurs in this saved-only audit.'],'new_Env_model_WM_Actor_GPU':False,'wall_seconds':time.perf_counter()-started}
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'source':pin(Path(__file__)),'new_model_or_GPU_execution':False},indent=2)+'\n')
        raise
