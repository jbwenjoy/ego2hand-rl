"""Arithmetic-only converter forensics. There is NO proxy scoring path here.

Original failed consumer/gates are untouched. The only native state conversion
is the frozen unbound reader helper; F64 calculations below are secondary
conditioning descriptors, never replacement converted states or eligibility.
"""
from __future__ import annotations
import hashlib,json,os,pickle,random,sys,time,zipfile
from pathlib import Path
import numpy as np
from . import penalty_observability105_v1 as original
from . import trained_bonus_analysis105_v1 as common

ROOT=common.ROOT
PREREG=ROOT/'protocols/full151105-converter-arithmetic-forensic-cpu-v1.json'
PREREG_SHA='06c5b055403962d963e8b8cc6f34ceba69399bef46afe0fb8582eeacb60daa13'
SCHEMA='full151105_converter_arithmetic_forensic_CPU_run:1'
ROLE='converter_arithmetic_forensics_only'
KEYS={'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
OWN={'consumer':ROOT/'src/full151_gradient/converter_forensic105_v1.py','CLI':ROOT/'scripts/diagnose_converter_forensic_cpu_v1.py','checker':ROOT/'scripts/check_converter_forensic_cpu_v1.py'}
require=common.require;pin=common.pin;checked=common.checked;read=common.read;write=common._write;npwrite=common._npwrite


def header(c):
    require(type(c) is dict and set(c)==KEYS and c['schema']==SCHEMA and c['role']==ROLE,'Exact forensic schema/role/keyset required')
    require(type(c['CPU_run_authorized']) is bool and type(c['cpu_threads']) is int and c['cpu_threads']==1,'Explicit CPU bool andthreads1 required')
    require(type(c['output']) is str and Path(c['output']).is_absolute(),'Absolute forensic output required')
    out=Path(c['output']).resolve();require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_converter_forensic_CPU_') and not out.exists(),'Fresh isolated forensic output required')
    require(type(c['source_pins']) is dict and c['source_pins'],'Source closure required')


def required_sources(p):
    parent=read(checked(p['parent_protocol']));result=original.required_sources(parent)
    case=read(checked(p['failed_config']))
    result.update({'failed_case__'+k:v for k,v in case['source_pins'].items()})
    result.update({k:pin(v) for k,v in OWN.items()});result['forensic_prereg_writer']=p['writer']
    return result


def declaration():
    require('torch' not in sys.modules and 'warp' not in sys.modules,'NoTorch/Warp default required')
    require(pin(PREREG)['sha256']==PREREG_SHA,'Forensic prereg changed')
    return {'schema':'full151105_converter_forensic_metadata:1','status':'DECLARED_ARITHMETIC_ONLY_NOT_EXECUTED','protocol':pin(PREREG),
      'runplan_schema':SCHEMA,'role':ROLE,'keys':sorted(KEYS),'source_pins':{k:pin(v) for k,v in OWN.items()},
      'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'MJB_loads':0,'physical_state_conversions':0,'scoring_calls':0}


def metadata_preflight(config_path):
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Fresh metadata process only')
    c=read(config_path);header(c);require(common.same_pin(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Fixed forensic condition required')
    p=read(checked(c['protocol']));require(p['schema']=='full151105_converter_arithmetic_forensic_protocol:1' and p['proxy_scoring_authorized'] is False,'Forensic-only condition required')
    parent=read(checked(p['parent_protocol']));require(pin(checked(p['parent_protocol']))['sha256']==original.PREREG_SHA,'Parent1ec18 required')
    failure=read(checked(p['failed_record']));require(failure['completed_scored_rows']==0 and failure['counts']['objective_factory']['attempted']==0 and failure['counts']['native_state_conversion']['returned']==1 and common.same_pin(failure['config'],p['failed_config']),'Original failure provenance changed')
    for value in parent['inputs'].values():checked(value)
    have={(str(checked(v)).casefold(),v['sha256']) for v in c['source_pins'].values()}
    require(all((str(checked(v)).casefold(),v['sha256']) in have for v in required_sources(p).values()),'Forensic source closure missing')
    ready=read(checked(c['cpu_ready']));proof=read(checked(ready['cpu_report']));common.all_checks(proof)
    require(ready['schema']=='full151105_converter_forensic_CPU_READY:1' and proof['schema']=='full151105_converter_forensic_CPU_proof:1' and common.same_pin(ready['protocol'],c['protocol']),'Wrong forensic fixture READY')
    for k,v in OWN.items():require(common.same_pin(ready['source_pins'][k],pin(v)) and common.same_pin(proof['source_pins'][k],pin(v)),'Forensic proof/source mismatch')
    require(proof['actual_physical_state_conversions']==proof['actual_MJB_or_objective_or_model_loads']==0,'Fixture not actual scientific run')
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Metadata imported backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'parent':parent,'status':'PASS_METADATA_ARITHMETIC_ONLY_NOT_EXECUTED','Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'scoring_calls':0}


def old_diagnostics(actual,reference,state58=False):
    """Exact old bound/bit flags, collected without raising on numeric failure."""
    a,b=np.asarray(actual),np.asarray(reference)
    require(a.shape==b.shape and a.ndim==2 and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Finite same-shapeFP32 diagnostic required')
    delta=a.astype(np.float64)-b.astype(np.float64);bound=8*2**-23*np.maximum(1.,np.maximum(abs(a.astype(float)),abs(b.astype(float))))
    flags=abs(delta)<=bound
    if state58:
        require(a.shape[1]==58,'state58 diagnostic expected')
        flags[:,:40]=a[:,:40].view(np.uint32)==b[:,:40].view(np.uint32)
    return {'actual':a.copy(),'reference':b.copy(),'difference_F64':delta,'old_bound_F64':bound,'old_flags':flags,'old_all_pass':bool(flags.all())}


def conditioning_shadow(q,v,pp,R,mapping):
    """Pure F64 SECONDARY arithmetic: no native state method/FK/query call."""
    q,v,pp,R=(np.asarray(a,np.float64) for a in (q,v,pp,R));n=len(q);oq=mapping['object_qpos_start'];ov=mapping['object_dof_start']
    require(q.shape==(n,27) and v.shape==(n,26) and pp.shape==(n,3) and R.shape==(n,3,3) and all(np.isfinite(a).all() for a in (q,v,pp,R)),'Conditioning input ABI')
    quat=q[:,oq+3:oq+7];norm=np.linalg.norm(quat,axis=-1,keepdims=True);require(np.all(norm>0),'Invalid raw quat')
    unit=quat/norm;w,x,y,z=unit.T
    objR=np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(n,3,3)
    inv=R.swapaxes(-1,-2);position=q[:,oq:oq+3];dp=position-pp;worldv=v[:,ov:ov+3];bodyw=v[:,ov+3:ov+6]
    mv=lambda A,x:np.einsum('bij,bj->bi',A,x)
    p=mv(inv,dp);r=inv@objR;vel=mv(inv,worldv);worldw=mv(objR,bodyw);omega=mv(inv,worldw)
    directq=q[:,mapping['hand_qpos']];directv=v[:,mapping['hand_dof']]
    shadow=np.concatenate((directq,directv,p,r.reshape(n,9),vel,omega),axis=-1)
    sum_p=mv(abs(inv),abs(dp));coordinate_scale=mv(abs(inv),abs(position)+abs(pp))
    sum_r=(abs(inv)@abs(objR)).reshape(n,9);sum_v=mv(abs(inv),abs(worldv));sum_w=mv(abs(inv),mv(abs(objR),abs(bodyw)))
    operands=np.concatenate((abs(directq),abs(directv),sum_p,sum_r,sum_v,sum_w),axis=-1)
    ratio=operands/np.maximum(abs(shadow),1e-30)
    require(np.isfinite(shadow).all() and np.isfinite(ratio).all(),'Nonfinite arithmetic conditioning')
    return {'F64_shadow_native_definition_NOT_ACCEPTED_STATE':shadow,'raw_object_quaternion':quat,'normalized_quaternion_F64':unit,'object_rotation_F64':objR,
      'inverse_actual_palm_rotation_F64':inv,'object_world_position_F64':position,'position_difference_F64':dp,'world_linear_velocity_F64':worldv,
      'body_angular_velocity_F64':bodyw,'world_angular_velocity_F64':worldw,'absolute_operand_sums_F64':operands,
      'position_coordinate_operand_scale_F64':coordinate_scale,'cancellation_ratios_DIAGNOSTIC_ONLY':ratio}


def block_summary(d,labels):
    blocks={'q':(0,20),'qd':(20,40),'position':(40,43),'rotation':(43,52),'linear_velocity':(52,55),'angular_velocity':(55,58)}
    result={}
    for name,(start,end) in blocks.items():
        errors=abs(d['difference_F64'][:,start:end]);where=np.unravel_index(np.argmax(errors),errors.shape);row,feature=map(int,where)
        result[name]={'failed_elements':int((~d['old_flags'][:,start:end]).sum()),'failed_rows':int((~d['old_flags'][:,start:end]).any(-1).sum()),
          'maximum_absolute_difference':float(errors[where]),'max_error_row':row,'max_error_feature':start+feature,'row_identity':labels[row]}
    return result


def run_forensic(config_path):
    start=time.perf_counter();p=metadata_preflight(config_path);c=p['config']
    require(c['CPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Explicit separate RootCPU forensic authorization required')
    out=Path(c['output']);out.mkdir();phase='archive';counts={'passive_MJB':{'attempted':0,'returned':0},'native_state_conversion':{'attempted':0,'returned':0}}
    def call(name,fn,*args,**kw):counts[name]['attempted']+=1;value=fn(*args,**kw);counts[name]['returned']+=1;return value
    try:
        write(out,'config_copy.json',c);write(out,'protocol.json',p['protocol']);write(out,'source_inventory.json',c['source_pins'])
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(c['source_pins'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        inputs=p['parent']['inputs'];phase='same_saved_preparation'
        samples=original._jsonl(checked(inputs['physical_samples']));rewards=original._jsonl(checked(inputs['reward_snapshots']));groups,reward_map=original.group_rows(samples,rewards)
        from . import analysis105_v1 as previous
        physical=checked(inputs['physical_samples']).parent;saved=read(checked(inputs['physical_saved_independent1762']))
        evidence={str(checked(v)).casefold():v for v in saved['evidence'].values()}
        def ev(path):return checked(evidence[str(Path(path).resolve(strict=True)).casefold()])
        previous._validate_capacity(read(ev(physical/'capacity_ledgers.json')))
        watch=Path(str(physical)+'_watchdog')
        previous._validate_lifecycle({'physical_run':str(physical),'artifact_pins':{'producer_config':inputs['physical_config']}},
          {'watch_state':read(ev(watch/'state.json')),'engineering_identity':read(ev(watch/'engineering_identity.json')),'post_terminal_CIM':read(ev(watch/'post_terminal_CIM.json')),'runtime':read(ev(physical/'runtime.json'))})
        import torch,mujoco
        require(not torch.cuda.is_initialized() and 'warp' not in sys.modules,'Fresh CPU forensic only')
        torch.set_num_threads(1)
        def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        before_rng=rng()
        from full151_native.bundle import load_bundle,FIELDS
        bundle=load_bundle(checked(inputs['full151_bundle']),expected_sha256=inputs['full151_bundle']['sha256'])
        for path in ('A0','A2'):
            with np.load(checked(inputs[path+'__expected_context_fields.npz']),allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
            with np.load(checked(inputs[path+'__actual_context_readbacks.npz']),allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
            require(set(expected)==set(FIELDS) and set(actual)=={stage+'__'+f for stage in ('after_copy','after_set_const_0') for f in FIELDS},'Actual8 source field set differs')
            for f in FIELDS:
                wanted=np.asarray(bundle['context'][f])[0];require(np.array_equal(expected[f],wanted),'Expected actual8 differs')
                for stage in ('after_copy','after_set_const_0'):
                    got=actual[stage+'__'+f];require(got.ndim==wanted.ndim+1 and got.shape[1:]==wanted.shape and np.array_equal(got,np.broadcast_to(wanted.astype(got.dtype),got.shape)),'Actual8 differs')
        model=call('passive_MJB',mujoco.MjModel.from_binary_path,str(checked(inputs['nominal_model'])));mapping=original._mapping(model);oq=mapping['object_qpos_start']
        q=original.exact_fp32([r['pre_qpos'] for r in samples],(1808,27),'allpreqpos');v=original.exact_fp32([r['pre_qvel'] for r in samples],(1808,26),'allpreqvel')
        pp=np.empty((1808,3),np.float32);R=np.empty((1808,3,3),np.float32);postq=[];postv=[];gpp=[];gR=[];refs=[];labels=[];lastids=[];source_palm={}
        with np.load(checked(inputs['native_data']),allow_pickle=False) as z:wrist=z['wrist_pose_w'].astype(np.float32)
        require(wrist.shape==(151,7) and np.array_equal(wrist,np.broadcast_to(wrist[:1],wrist.shape)),'Fixed native wrist changed')
        from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
        expectedR=quat_to_mat(torch.from_numpy(wrist[:1,3:].copy())).numpy()[0]
        for key,indices in groups:
            rw=reward_map[key];pos=original.exact_fp32(rw['state20_reader_palm_pos'],(1,3),'Readerpalm')[0];rot=original.exact_fp32(rw['state20_reader_palm_rot'],(1,3,3),'ReaderR')[0]
            original.arithmetic_check(pos,wrist[0,:3],'Reader/reference palm position');original.arithmetic_check(rot,expectedR,'Reader/reference palm rotation')
            if key[0] in source_palm:require(np.array_equal(source_palm[key[0]][0],pos) and np.array_equal(source_palm[key[0]][1],rot),'Reader frame changed within path')
            else:source_palm[key[0]]=(pos.copy(),rot.copy())
            pp[indices]=pos;R[indices]=rot;gpp.append(pos);gR.append(rot);last=indices[-1];lastids.append(last)
            postq.append(samples[last]['post_qpos']);postv.append(samples[last]['post_qvel']);refs.append(rw['physics']);labels.append({'path':key[0],'k':key[1],'branch_id':key[2],'last_sample_index':last})
        for path in source_palm:
            init=read(checked(inputs[path+'__initial_readback.json']));reset=read(checked(inputs[path+'__runtime_reset_readback.json']))
            for value in (init,reset):
                require(np.array_equal(np.asarray(value['mocap_pos'],np.float32).reshape(3),wrist[0,:3]) and np.array_equal(np.asarray(value['mocap_quat'],np.float32).reshape(4),wrist[0,3:]),'Startup/reset fixed parent differs')
        postq=original.exact_fp32(postq,(452,27),'endpointqpos');postv=original.exact_fp32(postv,(452,26),'endpointqvel');refs=original.exact_fp32(refs,(452,58),'knownpost20ref')
        allq=np.concatenate((q,postq));allv=np.concatenate((v,postv));allpp=np.concatenate((pp,np.asarray(gpp,np.float32)));allR=np.concatenate((R,np.asarray(gR,np.float32)))
        with np.load(checked(inputs['physical_arrays']),allow_pickle=False) as z:anchor=z['snapshot_x'].copy()
        anchor_indices=[];anchor_refs=[];anchor_labels=[]
        for key,indices in groups:
            if key[2] is not None or key[1] in (0,49,99,149):
                sid=('A0','A2').index(key[0])*4+(0,49,99,149).index(key[1]);anchor_indices.append(indices[0]);anchor_refs.append(anchor[sid]);anchor_labels.append({'path':key[0],'k':key[1],'branch_id':key[2],'pre_index':indices[0],'snapshot_id':sid})
        require(len(anchor_indices)==160 and len(allq)==2260,'Forensic exact2260/160 scope')
        solver_pose=original.exact_fp32([s['object_solver_pose'] for s in samples],(1808,7),'solverpose')
        require(np.all(abs(np.linalg.norm(solver_pose[:,3:].astype(float),axis=-1)-1)<=1e-5),'Malformed solverquat')
        phase='one_exact_native_conversion'
        converted=call('native_state_conversion',original.native_state58,allq,allv,allpp,allR,mapping,p['parent']['source_pins']['native_reader']).numpy().copy()
        # SAVE RAW FIRST: no endpoint-bound check is allowed to discard these.
        npwrite(out,'raw_conversion_all2260.npz',{'converted_native58':converted,'post20_reference':refs,'pre_qpos':q,'pre_qvel':v,'post20_qpos':postq,'post20_qvel':postv,
           'palm_pos_all2260':allpp,'palm_rot_all2260':allR,'object_solver_pose':solver_pose,'endpoint_last_sample_indices':np.asarray(lastids,np.int64),
           'anchor_pre_indices':np.asarray(anchor_indices,np.int64),'anchor_references':np.asarray(anchor_refs,np.float32),**{k:np.asarray(x) for k,x in mapping.items()}})
        write(out,'endpoint_identities.json',labels);write(out,'anchor_identities.json',anchor_labels)
        phase='all_old_rule_diagnostics_NO_EARLY_GATE'
        endpoint=old_diagnostics(converted[1808:],refs,True);leading=old_diagnostics(converted[anchor_indices],np.asarray(anchor_refs,np.float32),True)
        solver_R=quat_to_mat(torch.from_numpy(solver_pose[:,3:].copy())).numpy().reshape(1808,9)
        pre_R=quat_to_mat(torch.from_numpy(q[:,oq+3:oq+7].copy())).numpy().reshape(1808,9)
        pose=old_diagnostics(solver_pose[:,:3],q[:,oq:oq+3]);poseR=old_diagnostics(solver_R,pre_R)
        for name,value in [('endpoint452',endpoint),('leading160',leading),('solver_position1808',pose),('solver_rotation1808',poseR)]:
            npwrite(out,name+'_old_rule.npz',{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
        phase='secondary_conditioning_NOT_ELIGIBILITY'
        conditions=conditioning_shadow(allq,allv,allpp,allR,mapping)
        unit_CPU=torch.nn.functional.normalize(torch.from_numpy(allq[:,oq+3:oq+7].copy()),dim=-1).numpy()
        objR_CPU=quat_to_mat(torch.from_numpy(allq[:,oq+3:oq+7].copy())).numpy()
        conditions['normalized_quaternion_CPU_FP32']=unit_CPU;conditions['object_rotation_CPU_FP32']=objR_CPU
        conditions['converted_minus_F64_shadow']=converted.astype(float)-conditions['F64_shadow_native_definition_NOT_ACCEPTED_STATE']
        conditions['endpoint_GPU_reference_minus_F64_shadow']=refs.astype(float)-conditions['F64_shadow_native_definition_NOT_ACCEPTED_STATE'][1808:]
        npwrite(out,'conditioning_secondary_all2260.npz',conditions)
        summary={'endpoints':block_summary(endpoint,labels),'leadinganchors':block_summary(leading,anchor_labels),
          'new_replay_old_endpoint_rule_all_pass':endpoint['old_all_pass'],'new_replay_old_leading_rule_all_pass':leading['old_all_pass'],
          'solver_position_old_flags_all_pass':pose['old_all_pass'],'solver_rotation_old_flags_all_pass':poseR['old_all_pass'],
          'original_failed_gate_unchanged':'FAIL','replacement_gate_defined':False,'dense_scoring_eligibility':False}
        invariants={'RNG_unchanged':rng()==before_rng,'CUDA_uninitialized':not torch.cuda.is_initialized(),'Warp_not_imported':'warp' not in sys.modules,
           'binding_never_imported':'full151_intent.binding_v3' not in sys.modules,'WM_never_imported':'full151_native.world_model_checkpoint' not in sys.modules}
        require(all(invariants.values()),'Forensic read-only/source isolation failed')
        for value in inputs.values():checked(value)
        for value in c['source_pins'].values():checked(value)
        require(pin(config_path)==p['config_pin'],'Forensic config changed')
        report={'schema':'full151105_converter_arithmetic_forensic_result:1','status':'COMPLETED_ARITHMETIC_FORENSIC_NOT_DENSE_ELIGIBILITY',
          'config':p['config_pin'],'config_copy':pin(out/'config_copy.json'),'protocol':c['protocol'],'source_pins':c['source_pins'],'counts':counts,
          'converted_rows':2260,'endpoint_rows':452,'leading_anchor_comparisons':160,'solver_pose_comparisons':1808,'summary':summary,'invariants':invariants,
          'factory_calls':0,'proxy_or_WM_Actor_forward':0,'weights_AD_optimizer_Env_physics_nativeFK_collision_GPU':0,'old_failure_preserved':p['protocol']['failed_record'],
          'physical_gradient_qualified':False,'wholehiddenstate_qualified':False,'dense_scoring_eligibility':False,
          'scope':'SecondaryF64/operand sums/cancellation descriptors are not an error theorem or accepted converter replacement. Original8eps FAIL unchanged; frame/mapping/phase versus arithmetic cause not assumed. No factory/scoring code follows this function.',
          'elapsed_seconds':time.perf_counter()-start}
        write(out,'summary.json',summary);write(out,'report.json',report);write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        write(out,'failure.json',{'schema':'full151105_converter_arithmetic_forensic_failure:1','status':'FAILED_FORENSIC_PARTIAL_PRESERVED','error':repr(error),'phase':phase,'counts':counts,
          'factory_or_scoring_calls':0,'dense_scoring_eligibility':False,'old_failure_unchanged':p['protocol']['failed_record'],'config':p['config_pin']});raise
