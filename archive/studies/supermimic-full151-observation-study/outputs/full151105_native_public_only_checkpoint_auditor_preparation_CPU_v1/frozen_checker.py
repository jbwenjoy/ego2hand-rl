"""Two tagged synthetic saved-value cases. Root approval precedes execution.

No real artifacts, NN/optimizer construction, FW, AD, native Log or NPZ reads.
Sixteen tiny-shaped named tensors test group16/state14; actual native shapes
remain mandatory in production and are explicitly rejected for this fixture.
"""
import ast,copy,hashlib,importlib.util,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUDITOR=ROOT/'scripts/audit_native_public_only_checkpoint_saved_cpu_v1.py'
AUDITOR_SHA='be174f42df16034c3e5686e93357bd136371a0c64072db2de9b7be5826161a93'
OUT=ROOT/'outputs/full151105_native_public_only_checkpoint_auditor_preparation_CPU_v1'

def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def main():
    if OUT.exists():raise ValueError('Fresh synthetic proof output required')
    if pin(AUDITOR)['sha256']!=AUDITOR_SHA:raise ValueError('Reader source changed before first fixture')
    ast.parse(AUDITOR.read_text(encoding='utf-8'));ast.parse(Path(__file__).read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('native_only_saved_fixture_reader',AUDITOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    OUT.mkdir();sources={k:a.pin(v) for k,v in a.OWN.items()}
    for k in ('auditor','checker'):(OUT/('frozen_'+k+'.py')).write_bytes(Path(sources[k]['path']).read_bytes())
    write(OUT/'source_freeze.json',{'source_pins':sources,'before_synthetic':True,'actual_weights_or_labels_read':False})
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    torch.set_num_threads(4);a.require(not torch.cuda.is_initialized(),'CPU-only fixture')
    helper=a.B.load_helpers();checks={};load_attempts=[];original_load=torch.load
    def deny_load(*args,**kwargs):load_attempts.append(True);raise AssertionError('No torch.load allowed in these saved-value fixtures')
    torch.load=deny_load
    def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_CPU':torch.get_rng_state().clone(),'CUDA_initialized':False}
    before=rng()
    try:
        specs=tuple((name,(2,)) for name,_ in a.SPECS);names=[n for n,_ in specs];active=names[:14];inactive=names[14:]
        wm={n[3:]:torch.tensor([float(i+1),float(i+2)]) for i,n in enumerate(active)}
        wm.update({'norm.mean':torch.zeros(298),'norm.var':torch.ones(298),'norm.count':torch.tensor(42.,dtype=torch.float64),
            'delta_norm.mean':torch.zeros(52),'delta_norm.var':torch.ones(52),'delta_norm.count':torch.tensor(99.,dtype=torch.float64),'error_floor':torch.full((52,),.1)})
        head={n[9:]:torch.zeros(2) for n in inactive};host={'norm':{'eps':.01,'max_count':2e8,'_host_count':42.},'delta_norm':{'eps':.0001,'max_count':2e8,'_host_count':99.}}
        ctx={'mass_actual_FP32':torch.tensor(.03),'I_body_F64':torch.eye(3,dtype=torch.float64),'gravity_palm_F64':torch.zeros(3,dtype=torch.float64),
            'DeltaT_F64':torch.tensor(.02,dtype=torch.float64),'palm_rot_FP32':torch.eye(3),'effective_s52_FP32':torch.ones(52),
            'physical_xi46_FP32':torch.arange(46,dtype=torch.float32),'I_body_antisymmetry_maxabs_F64':torch.tensor(0.,dtype=torch.float64)}
        fixture_file=OUT/'synthetic_identity.json';write(fixture_file,{'scope':'SYNTHETIC_ONLY_NOT_EXECUTION_PROTOCOL'});pp=pin(fixture_file);rp=pin(a.RECIPE)
        own={name:dict(pp) for name in a.INPUTS}
        parent_keys={'target_report','target_arrays','target_saved8','target_source_inventory','initial_component','initial_validation','initial_saved_health29',
            'initial_parity_report','initial_parity_saved_review','initial_parity_baseline','component_ready','component_source_review','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7','target_identities'}
        parent={name:dict(pp) for name in parent_keys};source={'synthetic_reader':pin(AUDITOR),'synthetic_checker':pin(__file__)}
        y=torch.zeros((152,15));y[::2,0]=1.;y[1::3,10]=1.
        fits={'fit_input_digest':'1'*64,'fit_label_digest':'2'*64,'fit_self15_label_digest':helper.tensor_digest({'cached_self15_target_FP32':y},torch),'fit_combined_label_digest':'4'*64}
        common={'protocol':pp,'recipe':rp,'input_pins':own,'parent_training_inputs':parent,'parent_training_protocol':pp,'source_pins':source,**fits,
            'ordered_parameter_names':names,'active_gradient_parameter_names':active,'inactive_gradient_parameter_names':inactive,'zero_head_state_sha256':'5'*64}
        pending={'status':'PENDING_ITERATION1_NO_EXTRA_FW','source_baseline':parent['initial_parity_baseline'],'source_parity_report':parent['initial_parity_report']}
        initial={'schema':a.INITIAL_SCHEMA,'arm':a.ARM,'internal_wrapper_arm':a.INTERNAL_ARM,**copy.deepcopy(common),'source_parent':parent['initial_component'],
            'WM_state':copy.deepcopy(wm),'aux_head_state':copy.deepcopy(head),'physical_context_state':copy.deepcopy(ctx),'normalizer_host_configuration':copy.deepcopy(host),
            'RNG_this_run':rng(),'training_counts':dict.fromkeys(a.COUNTS,0),'cached_self15_target_FP32':y.clone(),'first_training_forward_parity':pending,
            'construction_receipt':{'raw_WM_unchanged':True,'disjoint_model_head_buffer_storage':True,'constructor_Python_TorchCPU_RNG_unchanged':True,
                'feature_hooks_independent':True,'raw_clone_state_identity':dict.fromkeys(a.B.ARMS,True)}}
        original={'schema':'full151105_WM_common_pre_component_calibration:1','WM':copy.deepcopy(wm),'normalizer_host_configuration':copy.deepcopy(host),'frozen_s52':ctx['effective_s52_FP32'].clone(),'physical_xi46':ctx['physical_xi46_FP32'].clone()}
        delta=(torch.arange(152*52,dtype=torch.float32).reshape(152,52)%17+1)/100
        full67=torch.zeros((152,67));full67[:,52:]=(torch.arange(15,dtype=torch.float32)[None]-7)/2;ell=full67[:,52:]
        squared=(delta/ctx['effective_s52_FP32']).square();point=squared.mean(-1)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(ell,y,reduction='none');selfsum=bce.sum(-1);pair=point+torch.tensor(.1)*selfsum
        scalars={'state':point.mean(),'self15':selfsum.mean(),'native_pair':pair.mean()}
        loss={'iteration':100,'tag':'PRE_OPTIMIZER_I','weights_after_updates':99,'native_D52_FP32':delta,'effective_s52_FP32':ctx['effective_s52_FP32'].clone(),
            'point_squared52_FP32':squared,'point_per_row_FP32':point,'self15_logits_FP32':ell.clone(),'self15_target_FP32':y.clone(),'self15_BCE_elements_FP32':bce,
            'self15_sum_per_row_FP32':selfsum,'native_pair_per_row_FP32':pair,'scalars':scalars}
        pre={name:(torch.full((2,),.25) if name in active else None) for name in names}
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(pre[n],2) for n in active]),2);coef=torch.clamp(1./(norm+1e-6),max=1.)
        post={name:(value*coef if value is not None else None) for name,value in pre.items()};postnorm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(post[n],2) for n in active]),2)
        gradient={'iteration':100,'tag':'PRE_OPTIMIZER_I','weights_after_updates':99,'parameter_names':names,'pre_clip':pre,'post_clip':post,'max_norm':1.,
            'pre_clip_returned_norm_FP32':norm,'post_clip_total_norm_FP32':postnorm}
        optimizer={'param_groups':[{'params':list(range(16)),'lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':0.,'foreach':False}],
            'state':{i:{'step':torch.tensor(100.),'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)} for i in range(14)}}
        final_wm=copy.deepcopy(wm)
        for name in active:final_wm[name[3:]]=final_wm[name[3:]]+.125
        payload={'schema':a.SCHEMA,'manifest_schema':a.MANIFEST_SCHEMA,'arm':a.ARM,'internal_wrapper_arm':a.INTERNAL_ARM,**copy.deepcopy(common),'iteration':100,
            'checkpoint_tag':'AFTER_OPTIMIZER_I','WM_state':final_wm,'aux_head_state':copy.deepcopy(head),'physical_context_state':copy.deepcopy(ctx),
            'normalizer_host_configuration':copy.deepcopy(host),'optimizer_state':optimizer,'counts':dict.fromkeys(a.COUNTS,100),'native_difference_calls':100,
            'RNG':copy.deepcopy(initial['RNG_this_run']),'gradient_snapshot':gradient,'loss_snapshot':loss,
            'first_training_forward_parity':{**pending,'status':'PASS_ITERATION1_NATIVE58_SELF15_AND_POSITIVE_ZERO14_BITS','training_iteration':1,'extra_validation_FW':0}}
        logs=[{'iteration':i,'arms':{a.ARM:{'counts':dict.fromkeys(a.COUNTS,i),'update_seconds':.001,'losses':{k:float(v) for k,v in scalars.items()}}}} for i in range(1,101)]
        def evaluate(value=payload,tagged=True):
            return a.validate_payload(original,initial,value,logs,100,pp,rp,own,source,parent,pp,fits,torch,helper,fixture_specs=specs if tagged else None)
        result,details=evaluate()
        checks['tagged_group16_state14_headNone_mixed_norm_PRE_AFTER_nine_operands_three_scalars']=all(result.values()) and details['fixture_only'] is True and details['actual16_14_contract'] is False and details['optimizer']['serialized_parameter_states']==14 and details['gradients']['inactive_head_None']==2
        def reject(call):
            try:call()
            except (ValueError,KeyError):return True
            return False
        mutations={}
        for name in ('head_zero_gradient','head_Adam_state','old_pair_namespace','extra_total_loss','swapped_own_parent_maps','wrong_PRE_phase','mixed_norm_count_cast'):
            bad=copy.deepcopy(payload)
            if name=='head_zero_gradient':bad['gradient_snapshot']['pre_clip'][inactive[0]]=torch.zeros(2)
            elif name=='head_Adam_state':bad['optimizer_state']['state'][14]=copy.deepcopy(bad['optimizer_state']['state'][0])
            elif name=='old_pair_namespace':bad['schema']='full151105_boundary_public_self15_WM_component:1'
            elif name=='extra_total_loss':bad['loss_snapshot']['scalars']['total']=bad['loss_snapshot']['scalars']['native_pair']
            elif name=='swapped_own_parent_maps':bad['input_pins']=copy.deepcopy(parent)
            elif name=='wrong_PRE_phase':bad['gradient_snapshot']['tag']='AFTER_OPTIMIZER_I'
            else:bad['WM_state']['delta_norm.count']=bad['WM_state']['delta_norm.count'].float()
            mutations[name]=reject(lambda bad=bad:evaluate(bad))
        mutations['Tiny_shapes_cannot_qualify_production_native16_14']=reject(lambda:evaluate(tagged=False))
        no_runtime=helper.raw_rng_comparison(before,rng(),torch)['all_raw_values_equal'] and not load_attempts and not torch.cuda.is_initialized()
        checks['meaningful_head_None_Adam_namespace_phase_loss_maps_dtype_and_Tiny_scope_rejections']=all(mutations.values()) and no_runtime
        a.require(not any(k.split('.')[0] in ('mujoco','warp','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules),'No native imports')
        a.require(all(checks.values()),'Two focused saved-value cases failed '+str(checks))
        a.require(all(a.pin(path)==sources[key] for key,path in a.OWN.items()),'Frozen sources changed during fixture')
        proof={'schema':'full151105_native_public_only_checkpoint_auditor_contract:1','status':'PASS_TAGGED_SYNTHETIC_NATIVE_ONLY_SAVED_VALUE_CONTRACT',
            'source_pins':sources,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'actual_real_loads_model_FW_AD_NPZ_physics':0,
            'fixture_only':True,'actual_native_parameter_shapes_qualified':False,'fixture_named_parameters':16,'fixture_active_states':14,
            'details':{'positive_saved_value_evidence':details,'rejections':mutations,'torch_load_calls':len(load_attempts),'raw_RNG_unchanged':no_runtime}}
        write(OUT/'report.json',proof);write(OUT/'READY.json',{'schema':'full151105_native_public_only_checkpoint_auditor_READY:1','source_pins':sources,'cpu_report':pin(OUT/'report.json'),
            'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'actual_checkpoint_health_qualified':False})
        print(json.dumps({'READY':pin(OUT/'READY.json'),'passed':proof['passed'],'total':proof['total']}));return 0
    except BaseException as error:
        if not (OUT/'failure.json').exists():write(OUT/'failure.json',{'error':repr(error),'source_pins':sources,'checks':checks,'automatic_retry':False,'actual_real_data_model_NPZ_FW_AD_physics':0})
        raise
    finally:torch.load=original_load

if __name__=='__main__':raise SystemExit(main())
