"""Saved-only fixed-visible58 offset versus finite action-response decomposition.

NumPy/stdlib only: no model, checkpoint, native model/state/geometry, objective,
goal, AD or physics API is imported or called. Baseline means are F64 statistics,
not new states or controls. Every saved branch/endpoint/pair is retained.
"""
from __future__ import annotations
import csv,hashlib,json,os,pickle,random,sys,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
PREREG=ROOT/'protocols/full151105_saved_action_response_CPU_prereg_v1.json'
PREREG_SHA='dc49c538cb996c46360917c3cf480f34ed410175b84af0e52ef82aad5be48221'
SCHEMA='full151105_saved_action_response_CPU_run:1'
ROLE='saved_fixed_state_offset_vs_finite_action_response'
KEYS={'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
OWN={'consumer':ROOT/'src/full151_gradient/action_response452105_v2.py','CLI':ROOT/'scripts/diagnose_full151_action_response_saved_cpu_v2.py','checker':ROOT/'scripts/check_full151_action_response_saved_cpu_v2.py'}
PHASES=(0,49,99,149);JOINTS=(0,4,8,12);EPS=(.002,.001)
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'R':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(path),'sha256':h.hexdigest()}


def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute(),'Exact absolute pin')
    sha=value['sha256'];require(type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'SHA256 format')
    p=Path(value['path']).resolve(strict=True);require(pin(p)['sha256']==sha,'Pinned bytes changed: '+str(p));return p


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def write(out,name,value):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def npwrite(out,name,value):
    require(not (out/name).exists(),'No overwrite');np.savez_compressed(out/name,**value)


def allchecks(report):
    c=report.get('checks');require(type(c) is dict and c and all(type(v) is bool and v for v in c.values()),'Strict alltrue saved proof')
    require(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed']==report['total']==len(c),'Saved proof counts')


def canonical_sources(values):
    if type(values) is dict:values=values.values()
    found={};spelling={};result=[]
    for v in values:
        require(type(v) is dict and set(v)=={'path','sha256'} and type(v['path']) is str and Path(v['path']).is_absolute(),'Source pin shape')
        key=(v['path'].casefold(),v['sha256'])
        if key in spelling:continue
        p=Path(v['path']).resolve(strict=True);s=p.stat();token=(s.st_dev,s.st_ino) if s.st_ino else ('resolved_path',str(p).casefold())
        if token in found:require(p.samefile(found[token]['path']) and v['sha256']==found[token]['sha256'],'Conflicting samefile pins')
        else:checked(v);found[token]=dict(v);result.append(dict(v))
        spelling[key]=token
    return {'file_%04d__%s'%(i,Path(v['path']).name):v for i,v in enumerate(result)}


def required_sources(p):return canonical_sources(list(p['source_pins'].values())+[pin(v) for v in OWN.values()])


def membership(table):
    result=set()
    for v in canonical_sources(table).values():
        p=checked(v);s=p.stat();result.add((s.st_dev,s.st_ino if s.st_ino else str(p).casefold(),v['sha256']))
    return result


def validate_header(c):
    require(type(c) is dict and set(c)==KEYS and c['schema']==SCHEMA and c['role']==ROLE,'Exact saved-response schema/role/keys')
    require(type(c['CPU_run_authorized']) is bool and type(c['cpu_threads']) is int and c['cpu_threads']==1,'Explicit CPU bool and fixed1 thread')
    require(type(c['source_pins']) is dict and c['source_pins'] and type(c['output']) is str and Path(c['output']).is_absolute(),'Source table/output required')
    out=Path(c['output']).resolve();require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_saved_action_response_CPU_') and not out.exists(),'Fresh scoped output')


def declaration():
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Fresh NumPy-only process')
    require(pin(PREREG)['sha256']==PREREG_SHA,'Prereg drift')
    return {'schema':'full151105_saved_action_response_declaration:1','status':'DECLARED_NOT_DECOMPOSED','runplan_schema':SCHEMA,'role':ROLE,'keys':sorted(KEYS),'protocol':pin(PREREG),
        'source_pins':{k:pin(v) for k,v in OWN.items()},'Torch_imported':False,'NPZ_opened':0,'model_weight_loads_FW_AD_MJB_Env_GPU_physics':0}


def parent_metadata(inputs):
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'No backend allowed')
    for v in inputs.values():checked(v)
    saved=read(checked(inputs['state452_saved_independent25']));allchecks(saved)
    require(saved['schema']=='full151105_state_prediction452_saved_independent:1' and saved['status']=='PASS_SAVED_ALL452_STATE_ERROR_REVIEW_NOT_MODEL_QUALIFICATION' and saved['engineering_qualified'] is True,'Wrong source452 proof')
    require(same(saved['config'],inputs['state452_config']) and same(saved['producer_report'],inputs['state452_report']),'State452 proof aliases')
    report=read(checked(inputs['state452_report']));run=checked(inputs['state452_report']).parent
    require(report['schema']=='full151105_state_prediction452_CPU_analysis:1' and report['status']=='COMPLETED_FIXED_A2_ALL452_STATE_ERROR_NOT_MODEL_QUALIFICATION' and report['prediction_rows']==452,'Incomplete source452')
    require(Path(saved['run_dir']).resolve(strict=True).samefile(run) and same(report['config'],inputs['state452_config']),'Source run/config route')
    output=read(run/'output_pins.json')
    for alias,name in (('state452_arrays','prediction_arrays.npz'),('state452_input_arrays','matched_input_arrays.npz'),('state452_identities','group_identities452.json')):
        require(same(inputs[alias],output[name]),'Source452 artifact alias: '+alias)
    physical=read(checked(inputs['physical_saved_independent1762']));allchecks(physical)
    require(same(physical['config'],inputs['physical_config']) and same(physical['producer_report'],inputs['physical_report']) and physical['completed_branches']==152,'Physical provenance')
    require(physical['physical_gradient_qualified'] is False and physical['whole_hidden_state_qualified'] is False,'Prior physical/hidden scope must stay unqualified')
    for alias in ('physical_config','physical_arrays','branch_index','snapshot_records'):
        require(same(report['input_pins'][alias],inputs[alias]),'Transitive physical source alias differs: '+alias)
    for path in ('A0','A2'):
        cm=read(checked(inputs[path+'_control_metadata']))
        for name in ('actuator_ids','joint_ids'):
            ids=cm[name];require(type(ids) is list and len(ids)==20 and all(type(v) is int and v>=0 for v in ids) and len(set(ids))==20,'Native ID JSON schema: '+path+'/'+name)
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Parent metadata imported backend')
    return {'status':'PASS_ACTUAL_PARENT_HASH_JSON_JOIN','state452_saved_review':inputs['state452_saved_independent25'],'physical_saved_review':inputs['physical_saved_independent1762'],'state452_run':str(run),'NPZ_decoded':0}


def metadata_preflight(config_path):
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'No backend allowed')
    c=read(config_path);validate_header(c);require(same(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Fixed prereg required')
    p=read(checked(c['protocol']));require(p['schema']=='full151105_saved_action_response_CPU_preregistration:1' and p['CPU_run_authorized'] is False,'Source-only prereg')
    for v in p['inputs'].values():checked(v)
    sources=canonical_sources(c['source_pins']);require(len(sources)==len(c['source_pins']) and membership(required_sources(p))<=membership(sources),'Oneactualfile/source membership')
    ready=read(checked(c['cpu_ready']));proof=read(checked(ready['cpu_report']));allchecks(proof)
    require(ready['schema']=='full151105_saved_action_response_CPU_READY:1' and proof['schema']=='full151105_saved_action_response_CPU_contract:1' and proof['status']=='PASS_SYNTHETIC_RESPONSE_PAIR_GUARDS_NO_MODEL','Wrong new proof scope')
    require(same(ready['protocol'],c['protocol']),'Ready protocol')
    for k,v in OWN.items():require(same(ready['source_pins'][k],pin(v)) and same(proof['source_pins'][k],pin(v)),'Executed source differs: '+k)
    require(proof['actual_saved_response_runs']==proof['actual_model_FW_AD_MJB_physics']==proof['actual_saved_array_reads']==0,'Fixture elevated')
    parents=parent_metadata(p['inputs'])
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Metadata imported backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'sources':sources,'parent_metadata':parents,'status':'PASS_HASH_JSON_SAVED_RESPONSE_ONLY','Torch_imported':False,'NPZ_opened':0,'decompositions':0}


def f32(a,shape,name):
    a=np.asarray(a);require(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'Finite FP32 '+name);return a


def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype==np.float32 and a.tobytes()==b.tobytes()


def json_control(value):
    source=np.asarray(value);require(source.shape==(20,) and source.dtype.kind in 'fiu' and np.isfinite(source).all(),'Source control20')
    out=source.astype(np.float32);require(np.array_equal(out.astype(np.float64),source.astype(np.float64)),'No silent control rounding');return out


def prepare_inputs(state,identities,rows,physical,control_meta):
    """Mechanical source joins and gates. No mean can hide a control/pre mismatch."""
    for name,shape in (('pre_x452_FP32',(452,58)),('predicted_x452_FP32',(452,58)),('target_x452_FP32',(452,58)),('actual_u452_FP32',(452,20))):f32(state[name],shape,name)
    require(type(identities) is list and len(identities)==452 and type(rows) is list and len(rows)==152,'Complete source row sets')
    for name,shape in (('snapshot_x',(8,58)),('snapshot_u',(8,20)),('branch_u',(152,20)),('direction__centers',(8,20)),
        ('direction__minus',(8,4,2,20)),('direction__plus',(8,4,2,20))):f32(physical[name],shape,name)
    for name in ('direction__d_eff','direction__center_offset'):
        require(physical[name].shape==(8,4,2,20) and physical[name].dtype==np.float64 and np.isfinite(physical[name]).all(),'Original F64 '+name)
    require(physical['direction__clipped'].shape==(8,4,2) and physical['direction__clipped'].dtype==np.bool_,'Original clipped flag ABI')
    by_branch={}
    for i,identity in enumerate(identities):
        require(type(identity['group_id']) is int and identity['group_id']==i,'Source group ID/order')
        bid=identity['branch_id']
        if bid is not None:
            require(type(bid) is int and 0<=bid<152 and bid not in by_branch,'Unique source branch identity');by_branch[bid]=i
    require(set(by_branch)==set(range(152)),'All152 restored rows required')
    indices=np.array([by_branch[i] for i in range(152)],np.int64).reshape(8,19)
    pre=state['pre_x452_FP32'][indices].copy();pred=state['predicted_x452_FP32'][indices].copy();target=state['target_x452_FP32'][indices].copy();u=state['actual_u452_FP32'][indices].copy()
    same40=pre[:,:,:40].view(np.uint32)==physical['snapshot_x'][:,None,:40].view(np.uint32)
    same18=pre[:,:,40:]==physical['snapshot_x'][:,None,40:]
    internal40=pre[:,:,:40].view(np.uint32)==pre[:,:1,:40].view(np.uint32);internal18=pre[:,:,40:]==pre[:,:1,40:]
    baseline_u=u[:,:3].view(np.uint32)==physical['snapshot_u'][:,None,:].view(np.uint32)
    branch_u=u.reshape(152,20).view(np.uint32)==physical['branch_u'].view(np.uint32)
    centers=physical['direction__centers'].view(np.uint32)==physical['snapshot_u'].view(np.uint32)
    nonj=np.ones((8,16,20),np.bool_);metadata_rows=[];native_ids=np.empty((8,4,2),np.int64)
    for s in range(8):
        path=('A0','A2')[s//4];k=PHASES[s%4];cm=control_meta[path]
        aids=np.asarray(cm['actuator_ids']);jids=np.asarray(cm['joint_ids'])
        require(aids.shape==jids.shape==(20,) and aids.dtype.kind in 'iu' and jids.dtype.kind in 'iu' and len(set(aids.tolist()))==len(set(jids.tolist()))==20 and (aids>=0).all() and (jids>=0).all(),'Saved native ID mapping20')
        for j,coordinate in enumerate(JOINTS):native_ids[s,j]=[int(aids[coordinate]),int(jids[coordinate])]
        for off in range(19):
            bid=s*19+off;row=rows[bid];identity=identities[int(indices[s,off])]
            base=off<3;j=None if base else (off-3)%8//2;ei=None if base else (off-3)//8;sign=0 if base else (-1,1)[(off-3)%2]
            wanted={'row_id':bid,'snapshot_id':s,'path':path,'k':k,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else EPS[ei],
                'sign':sign,'baseline_repeat':off if base else None,'joint_index':None if base else JOINTS[j],'type':'baseline' if base else 'direction'}
            require(all(type(row.get(key)) is type(value) and row[key]==value for key,value in wanted.items()),'Original branch metadata/order differs: '+str(bid))
            require(identity['path']==path and identity['k']==k and identity['branch_id']==bid and identity['mode']==wanted['type'],'State452/original branch join differs')
            require(bits(json_control(row['actualu']),physical['branch_u'][bid]),'Original JSON/physical control bytes differ')
            metadata_rows.append({**wanted,'group_id':int(indices[s,off]),'native_actuator_id':None if base else int(aids[JOINTS[j]]),'native_joint_id':None if base else int(jids[JOINTS[j]])})
            if not base:
                mask=np.arange(20)!=JOINTS[j];nonj[s,off-3,mask]=u[s,off,mask].view(np.uint32)==physical['snapshot_u'][s,mask].view(np.uint32)
    actual_minus=np.empty((8,4,2,20),np.float32);actual_plus=np.empty_like(actual_minus)
    for j in range(4):
        for ei in range(2):
            off=3+8*ei+2*j;actual_minus[:,j,ei]=u[:,off];actual_plus[:,j,ei]=u[:,off+1]
    endpoint_minus=actual_minus.view(np.uint32)==physical['direction__minus'].view(np.uint32);endpoint_plus=actual_plus.view(np.uint32)==physical['direction__plus'].view(np.uint32)
    delta=actual_plus.astype(np.float64)-actual_minus.astype(np.float64);legacy=delta/(2*np.asarray(EPS,np.float64)[None,None,:,None])
    coordinate_delta=np.stack([delta[:,j,:,coordinate] for j,coordinate in enumerate(JOINTS)],axis=1)
    center_offset=(actual_plus.astype(np.float64)+actual_minus.astype(np.float64))*.5-physical['snapshot_u'][:,None,None,:].astype(np.float64)
    legacy_equal=legacy==physical['direction__d_eff'];offset_equal=center_offset==physical['direction__center_offset']
    calculated_clip=np.empty((8,4,2),np.bool_)
    for s in range(8):
        for j,coordinate in enumerate(JOINTS):
            unit=np.eye(20,dtype=np.float32)[coordinate]
            for ei,eps in enumerate(EPS):
                center=physical['snapshot_u'][s]
                calculated_clip[s,j,ei]=not (np.array_equal(actual_plus[s,j,ei],center+np.float32(eps)*unit) and np.array_equal(actual_minus[s,j,ei],center-np.float32(eps)*unit))
    guards={'same_original_snapshot_bits40':same40,'same_original_snapshot_numeric18':same18,'internally_same_bits40':internal40,'internally_same_numeric18':internal18,
        'baseline_u_bitexact_original':baseline_u,'all_actual_u_bitexact_branch_authority':branch_u,'direction_centers_bitexact_original':centers,'perturbation_nonj_bitexact_original':nonj,
        'minus_endpoint_bitexact':endpoint_minus,'plus_endpoint_bitexact':endpoint_plus,'legacy_d_eff_numericexact':legacy_equal,'center_offset_numericexact':offset_equal,'clipping_flag_exact':calculated_clip==physical['direction__clipped'],
        'source_plus_not_less_than_minus_coordinate':coordinate_delta>=0}
    raw={'pre_visible152_FP32':pre,'predicted_y152_FP32':pred,'target_y152_FP32':target,'actual_u152_FP32':u,'branch_group_indices':indices,'original_snapshot_visible58_FP32':physical['snapshot_x'].copy(),
        'original_snapshot_u_FP32':physical['snapshot_u'].copy(),'actual_u_minus_rad_FP32':actual_minus,'actual_u_plus_rad_FP32':actual_plus,'legacy_d_eff_dimensionless':physical['direction__d_eff'].copy(),
        'center_offset_rad':physical['direction__center_offset'].copy(),'legacy_clipped':physical['direction__clipped'].copy(),'native_actuator_joint_ids':native_ids}
    return raw,guards,metadata_rows


def decompose(raw,guards):
    require(all(np.asarray(v).dtype==np.bool_ and np.asarray(v).all() for v in guards.values()),'BLOCKED fixed visible58/control/source precondition; no partial passed-snapshot subset')
    p,t,u=raw['predicted_y152_FP32'],raw['target_y152_FP32'],raw['actual_u152_FP32']
    pm=p[:,:3].astype(np.float64).mean(1);tm=t[:,:3].astype(np.float64).mean(1);um=u[:,:3].astype(np.float64).mean(1)
    pr=p[:,3:].astype(np.float64)-pm[:,None,:];tr=t[:,3:].astype(np.float64)-tm[:,None,:]
    result={'baseline_predicted_mean_F64':pm,'baseline_target_mean_F64':tm,'baseline_u_mean_F64':um,'constant_baseline_error_F64':pm-tm,
        'centered_model_response_F64':pr,'centered_physical_response_F64':tr,'centered_response_error_F64':pr-tr,'perturbation_control_offset_rad_F64':u[:,3:].astype(np.float64)-um[:,None,:],
        'perturbation_control_offset_from_original_u0_rad_F64':u[:,3:].astype(np.float64)-raw['original_snapshot_u_FP32'][:,None,:].astype(np.float64)}
    for name,value in (('model',p[:,:3]),('physical',t[:,:3]),('u',u[:,:3])):
        v=value.astype(np.float64);center=v.mean(1);deviation=v-center[:,None,:]
        result['baseline_'+name+'_range_F64']=v.max(1)-v.min(1)
        result['baseline_'+name+'_centered_variance_F64']=(deviation*deviation).mean(1)
        result['baseline_'+name+'_maxabs_centered_deviation_F64']=abs(deviation).max(1)
        result['baseline_'+name+'_replicates_FP32']=value.copy()
        result['baseline_'+name+'_bits_uint32']=value.view(np.uint32).copy()
    delta=raw['actual_u_plus_rad_FP32'].astype(np.float64)-raw['actual_u_minus_rad_FP32'].astype(np.float64)
    den=np.stack([delta[:,j,:,coordinate] for j,coordinate in enumerate(JOINTS)],axis=1);defined=den!=0
    model_minus=np.empty((8,4,2,58),np.float32);model_plus=np.empty_like(model_minus);physical_minus=np.empty_like(model_minus);physical_plus=np.empty_like(model_minus)
    for j in range(4):
        for ei in range(2):
            off=3+8*ei+2*j;model_minus[:,j,ei]=p[:,off];model_plus[:,j,ei]=p[:,off+1];physical_minus[:,j,ei]=t[:,off];physical_plus[:,j,ei]=t[:,off+1]
    pn=model_plus.astype(np.float64)-model_minus.astype(np.float64);tn=physical_plus.astype(np.float64)-physical_minus.astype(np.float64)
    ps=np.full_like(pn,np.nan);ts=np.full_like(tn,np.nan)
    np.divide(pn,den[...,None],out=ps,where=defined[...,None]);np.divide(tn,den[...,None],out=ts,where=defined[...,None])
    plus_distance=raw['actual_u_plus_rad_FP32'].astype(np.float64)-um[:,None,None,:];minus_distance=um[:,None,None,:]-raw['actual_u_minus_rad_FP32'].astype(np.float64)
    asym=np.stack([plus_distance[:,j,:,coordinate]!=minus_distance[:,j,:,coordinate] for j,coordinate in enumerate(JOINTS)],axis=1)
    hplus=np.stack([plus_distance[:,j,:,coordinate] for j,coordinate in enumerate(JOINTS)],axis=1)
    hminus=np.stack([minus_distance[:,j,:,coordinate] for j,coordinate in enumerate(JOINTS)],axis=1)
    result.update(actual_delta_u_rad=delta,actual_halfstep_u_rad=delta*.5,actual_coordinate_denominator_rad=den,secant_defined_mask=defined,secant_undefined_mask=~defined,
        model_pair_numerator_F64=pn,physical_pair_numerator_F64=tn,model_coordinate_secant_F64=ps,physical_coordinate_secant_F64=ts,coordinate_secant_error_F64=ps-ts,
        model_pair_minus_FP32=model_minus,model_pair_plus_FP32=model_plus,physical_pair_minus_FP32=physical_minus,physical_pair_plus_FP32=physical_plus,
        model_pair_numeric_zero=pn==0,physical_pair_numeric_zero=tn==0,
        model_pair_bitexact_zero=model_plus.view(np.uint32)==model_minus.view(np.uint32),physical_pair_bitexact_zero=physical_plus.view(np.uint32)==physical_minus.view(np.uint32),
        plus_distance_from_baseline_rad=plus_distance,minus_distance_from_baseline_rad=minus_distance,asymmetric_about_baseline=asym,
        hplus_coordinate_rad=hplus,hminus_coordinate_rad=hminus,hplus_minus_hminus_coordinate_rad=hplus-hminus,
        model_secant_finite_mask=np.isfinite(ps),physical_secant_finite_mask=np.isfinite(ts))
    # Different F64 association may round differently. Preserve, never gate it.
    result['centered_residual_identity_roundoff_F64']=(pr-tr)-((p[:,3:].astype(np.float64)-t[:,3:].astype(np.float64))-(pm-tm)[:,None,:])
    return result


def statistics(result):
    defined=result['secant_defined_mask'];stats={}
    for name,(lo,hi,unit) in BLOCKS.items():
        def metric(v):
            a=np.asarray(v)
            if not a.size:return {'elements':0,'component_RMS':None,'MAE':None,'maximum_absolute':None,'per_feature_RMS':[None]*a.shape[-1],'per_feature_MAE':[None]*a.shape[-1],'per_feature_maxabs':[None]*a.shape[-1]}
            axes=tuple(range(a.ndim-1));value={'elements':int(a.size),'component_RMS':float(np.sqrt(np.mean(a*a))),'MAE':float(abs(a).mean()),'maximum_absolute':float(abs(a).max()),
                'per_feature_RMS':np.sqrt(np.mean(a*a,axis=axes)).tolist(),'per_feature_MAE':abs(a).mean(axis=axes).tolist(),'per_feature_maxabs':abs(a).max(axis=axes).tolist()}
            if name=='R':value['raw9_Frobenius_RMS']=float(np.sqrt(np.mean(np.sum(a*a,axis=-1))))
            return value
        values={'state_units':unit,'coordinate_secant_units':unit+'/control_rad',
            'constant_baseline_error':metric(result['constant_baseline_error_F64'][:,lo:hi]),
            'model_centered_response':metric(result['centered_model_response_F64'][:,:,lo:hi]),'physical_centered_response':metric(result['centered_physical_response_F64'][:,:,lo:hi]),
            'centered_response_error':metric(result['centered_response_error_F64'][:,:,lo:hi]),
            'model_coordinate_secant_DEFINED_ONLY':metric(result['model_coordinate_secant_F64'][defined][:,lo:hi]),
            'physical_coordinate_secant_DEFINED_ONLY':metric(result['physical_coordinate_secant_F64'][defined][:,lo:hi]),
            'coordinate_secant_error_DEFINED_ONLY':metric(result['coordinate_secant_error_F64'][defined][:,lo:hi]),
            'defined_pair_count':int(defined.sum()),'undefined_pairs_kept':int((~defined).sum()),
            'model_encoded_numeric_zero_component_count':int(result['model_pair_numeric_zero'][...,lo:hi].sum()),'physical_encoded_numeric_zero_component_count':int(result['physical_pair_numeric_zero'][...,lo:hi].sum()),
            'baseline_physical_range_maximum':float(result['baseline_physical_range_F64'][:,lo:hi].max()),'baseline_model_range_maximum':float(result['baseline_model_range_F64'][:,lo:hi].max())}
        stats[name]=values
    return stats


def per_feature_statistics(result):
    """All58 coordinates separately; no mixed-unit reduction across features."""
    defined=result['secant_defined_mask']
    sources={'constant_baseline_error':result['constant_baseline_error_F64'],
        'model_centered_response':result['centered_model_response_F64'],'physical_centered_response':result['centered_physical_response_F64'],
        'centered_response_error':result['centered_response_error_F64'],
        'model_coordinate_secant_DEFINED_ONLY':result['model_coordinate_secant_F64'][defined],
        'physical_coordinate_secant_DEFINED_ONLY':result['physical_coordinate_secant_F64'][defined],
        'coordinate_secant_error_DEFINED_ONLY':result['coordinate_secant_error_F64'][defined]}
    output={'feature_indices':list(range(58)),'state_units':[unit for lo,hi,unit in BLOCKS.values() for _ in range(hi-lo)],
        'defined_pair_count':int(defined.sum()),'undefined_pair_count_kept':int((~defined).sum()),'statistics':{}}
    for name,a in sources.items():
        axes=tuple(range(a.ndim-1));count=int(np.prod(a.shape[:-1]))
        output['statistics'][name]={'samples_per_feature':count,
            'RMS':np.sqrt(np.mean(a*a,axis=axes)).tolist() if count else [None]*58,
            'MAE':abs(a).mean(axis=axes).tolist() if count else [None]*58,
            'maximum_absolute':abs(a).max(axis=axes).tolist() if count else [None]*58}
    return output


def run_analysis(config_path):
    started=time.perf_counter();m=metadata_preflight(config_path);c=m['config'];require(c['CPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Explicit fresh saved-only CPU authorization')
    out=Path(c['output']);out.mkdir();phase='archive';p=m['protocol'];inputs=p['inputs']
    try:
        write(out,'config.json',c);write(out,'protocol.json',p);write(out,'source_inventory.json',m['sources'])
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(m['sources'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        phase='read_saved_rows_and_guard'
        keys=('pre_x452_FP32','predicted_x452_FP32','target_x452_FP32','actual_u452_FP32')
        with np.load(checked(inputs['state452_arrays']),allow_pickle=False) as z:state={k:z[k].copy() for k in keys}
        physical_keys=('snapshot_x','snapshot_u','branch_u','direction__centers','direction__minus','direction__plus','direction__d_eff','direction__center_offset','direction__clipped')
        with np.load(checked(inputs['physical_arrays']),allow_pickle=False) as z:physical={k:z[k].copy() for k in physical_keys}
        npwrite(out,'selected_source452_raw.npz',state);npwrite(out,'selected_physical_direction_raw.npz',physical)
        identities=read(checked(inputs['state452_identities']));rows=read(checked(inputs['branch_index']));cm={path:read(checked(inputs[path+'_control_metadata'])) for path in ('A0','A2')}
        raw,guards,metadata=prepare_inputs(state,identities,rows,physical,cm)
        npwrite(out,'all152_branch_raw.npz',raw);npwrite(out,'visible_and_control_guard_masks.npz',guards);write(out,'all152_branch_metadata.json',metadata)
        passed=all(v.all() for v in guards.values());write(out,'guard_summary.json',{k:{'all_passed':bool(v.all()),'failed_elements':int((~v).sum()),'first_failed_indices':np.argwhere(~v)[:20].tolist()} for k,v in guards.items()})
        if not passed:
            report={'schema':'full151105_saved_action_response_CPU_analysis:1','status':'BLOCKED_FIXED_VISIBLE_STATE_OR_CONTROL_MAPPING_NO_REPAIR','fixed_visible_state_interpretation_permitted':False,
                'config':m['config_pin'],'protocol':c['protocol'],'source_pins':m['sources'],'input_pins':inputs,'all152_branches_saved':True,'responses_computed':False,
                'model_loads_forward_AD_MJB_Env_GPU_physics':0,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'elapsed_seconds':time.perf_counter()-started}
            write(out,'report.json',report);return report
        phase='F64_baseline_response_secants'
        before={k:pin(v['path']) for k,v in inputs.items()};rng_before=pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)
        result=decompose(raw,guards);stats=statistics(result);features=per_feature_statistics(result)
        npwrite(out,'decomposition_arrays.npz',result);write(out,'statistics.json',stats);write(out,'per_feature58_statistics.json',features)
        pair_records=[]
        for s in range(8):
            for j,coordinate in enumerate(JOINTS):
                for ei,eps in enumerate(EPS):
                    off=3+8*ei+2*j;defined=bool(result['secant_defined_mask'][s,j,ei])
                    pair_records.append({'pair_id':len(pair_records),'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':coordinate,'native_actuator_id':int(raw['native_actuator_joint_ids'][s,j,0]),
                        'native_joint_id':int(raw['native_actuator_joint_ids'][s,j,1]),
                        'epsilon_index':ei,'requested_epsilon_rad':eps,'minus_branch_id':s*19+off,'plus_branch_id':s*19+off+1,
                        'actual_coordinate_denominator_rad':float(result['actual_coordinate_denominator_rad'][s,j,ei]),'secant_defined':defined,
                        'hplus_coordinate_rad':float(result['hplus_coordinate_rad'][s,j,ei]),'hminus_coordinate_rad':float(result['hminus_coordinate_rad'][s,j,ei]),'hplus_minus_hminus_coordinate_rad':float(result['hplus_minus_hminus_coordinate_rad'][s,j,ei]),
                        'legacy_clipped':bool(raw['legacy_clipped'][s,j,ei]),'asymmetric_about_baseline':bool(result['asymmetric_about_baseline'][s,j,ei]),'status':'DEFINED' if defined else 'UNDEFINED_ZERO_ACTUAL_COORDINATE_DENOMINATOR'})
        write(out,'all64_pair_records.json',pair_records)
        require(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)==rng_before,'Saved-only RNG changed')
        require({k:pin(v['path']) for k,v in inputs.items()}==before,'Source input bytes changed')
        for v in inputs.values():checked(v)
        for v in m['sources'].values():checked(v)
        require(same(pin(config_path),m['config_pin']),'Actual config changed')
        report={'schema':'full151105_saved_action_response_CPU_analysis:1','status':'COMPLETED_SAVED_OFFSET_AND_FINITE_ACTION_RESPONSE_NOT_JACOBIAN',
            'fixed_visible_state_interpretation_permitted':True,'config':m['config_pin'],'protocol':c['protocol'],'source_pins':m['sources'],'input_pins':inputs,
            'counts':{'snapshots':8,'all_branches':152,'baseline_rows':24,'perturbation_endpoints':128,'all_epsilon_pairs':64,'defined_pairs':int(result['secant_defined_mask'].sum()),'undefined_pairs_kept':int(result['secant_undefined_mask'].sum())},
            'statistics':stats,'per_feature58_statistics':features,'invariants':{'input_files_unchanged':True,'RNG_unchanged':True,'Torch_Warp_MuJoCo_not_imported':all(k not in sys.modules for k in ('torch','warp','mujoco'))},
            'model_weight_loads_forward_AD_MJB_Env_GPU_physics':0,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,
            'scope':'Same recorded visible58 and absoluteu controls only. Baseline F64means are statistics, not new physical states. Empirical3baseline range/variance is not a true noise bound. CPUshadow one-step finite response, not trueJacobian/HO or historical GPUAMP gradient. No goal/taskscore/objective or stable9 filtering.',
            'elapsed_seconds':time.perf_counter()-started}
        require(report['invariants']['Torch_Warp_MuJoCo_not_imported'],'Backend imported')
        write(out,'report.json',report);write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        failure={'schema':'full151105_saved_action_response_CPU_failure:1','status':'FAILED_SAVED_INPUT_OR_CONTRACT_NO_REPAIR','phase':phase,'error':repr(error),'config':m['config_pin'],'model_weight_loads_forward_AD_MJB_Env_GPU_physics':0}
        write(out,'failure.json',failure)
        if not (out/'report.json').exists():write(out,'report.json',{'schema':'full151105_saved_action_response_CPU_analysis:1','status':'BLOCKED_FIXED_VISIBLE_STATE_OR_CONTROL_MAPPING_NO_REPAIR',
            'fixed_visible_state_interpretation_permitted':False,'source_contract_failure':failure,'responses_scientifically_interpretable':False,'config':m['config_pin'],'protocol':c['protocol'],'model_weight_loads_forward_AD_MJB_Env_GPU_physics':0})
        raise
