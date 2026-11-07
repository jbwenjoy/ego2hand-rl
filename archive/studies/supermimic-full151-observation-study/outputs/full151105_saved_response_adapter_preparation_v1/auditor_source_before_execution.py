"""Independent saved response audit; no model/native/API re-execution.

All raw152,128responses,64pairs are retained. Original FP32 clipping uses
numeric array equality, baseline/non-j/q/qd predicates retain exact bits.
F64 means are statistical references, not new states/actions. No new floor,
stable filter, score, mixed-unit58 norm or Jacobian qualification is introduced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/saved_action_response_independent_math_v1.py'
MATH_SHA='72ad09cc29c0877104548dcd127fc0cda47e6ed69986bd8649af96d1566d9cc8'
CORE_SHA='17cb3805ab0e4906f0ca73ff6f289b9bdf525aafd22000af0498387cbac84af3'
PROTOCOL_SHA='dc49c538cb996c46360917c3cf480f34ed410175b84af0e52ef82aad5be48221'
PHASES=(0,49,99,149)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def pin(p):
    return {'path': str(Path(p).resolve(strict=True)), 'sha256': sha(p)}

def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if sha(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path

def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])

def dump(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def module(path, expected):
    if sha(path) != expected:
        raise ValueError('Independent math changed')
    spec = importlib.util.spec_from_file_location('independent_'+path.stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)


def strict_json(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return set(a)==set(b) and all(strict_json(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(strict_json(x,y) for x,y in zip(a,b))
    return a==b


def native_pair_order(values,identities):
    """Explicit (snapshot,j,epsilon) bijection, never a blind reshape."""
    a=np.asarray(values)
    if len(identities)!=64 or a.shape[0]!=64: raise ValueError('All64 independent pairs required')
    out=np.empty((8,4,2)+a.shape[1:],dtype=a.dtype);seen=set()
    for i,item in enumerate(identities):
        key=(item['snapshot_id'],item['joint_ordinal'],item['epsilon_index'])
        if key in seen or key not in {(s,j,e) for s in range(8) for j in range(4) for e in range(2)}: raise ValueError('Pair identity duplicate/range')
        if item['pair_id']!=i or item['joint_index']!=(0,4,8,12)[key[1]] or item['requested_epsilon_rad']!=(.002,.001)[key[2]]: raise ValueError('Independent pair metadata')
        off=3+key[2]*8+key[1]*2
        if item['minus_branch_id']!=key[0]*19+off or item['plus_branch_id']!=key[0]*19+off+1: raise ValueError('Pair endpoint order')
        seen.add(key);out[key]=a[i]
    if len(seen)!=64: raise ValueError('Missing pairs')
    return out


def expected_decomposition(raw,math):
    p,t,u=raw['predicted_y152_FP32'],raw['target_y152_FP32'],raw['actual_u152_FP32']
    result=math.decompose(raw['pre_visible152_FP32'],t,p,u,raw['original_snapshot_visible58_FP32'],raw['original_snapshot_u_FP32'],
        {'minus':raw['actual_u_minus_rad_FP32'],'plus':raw['actual_u_plus_rad_FP32'],'d_eff':raw['legacy_d_eff_dimensionless'],
         'center_offset':raw['center_offset_rad'],'clipped':raw['legacy_clipped']})
    if not result['fixed_visible_state_interpretation_allowed']: return result,None
    a=result['arrays'];pair=lambda key:native_pair_order(a[key],result['pair_identities'])
    pm,tm,um=a['baseline_predicted_mean_F64'],a['baseline_physical_mean_F64'],a['baseline_control_mean_F64']
    e={'baseline_predicted_mean_F64':pm,'baseline_target_mean_F64':tm,'baseline_u_mean_F64':um,'constant_baseline_error_F64':a['baseline_bias_F64'],
       'centered_model_response_F64':a['predicted_response128_F64'].reshape(8,16,58),'centered_physical_response_F64':a['physical_response128_F64'].reshape(8,16,58),
       'centered_response_error_F64':a['response_residual128_F64'].reshape(8,16,58),
       'perturbation_control_offset_rad_F64':a['actual_du_from_baseline_mean128_rad'].reshape(8,16,20),
       'perturbation_control_offset_from_original_u0_rad_F64':a['actual_du_from_original_u0_128_rad'].reshape(8,16,20)}
    for name,kind,value in [('model','predicted',p[:,:3]),('physical','physical',t[:,:3]),('u','control',u[:,:3])]:
        n=result['baseline_noise'][kind]
        for dest,src in [('range','range_F64'),('centered_variance','population_variance_F64'),('maxabs_centered_deviation','maximum_absolute_centered_F64')]:
            e['baseline_'+name+'_'+dest+'_F64']=n[src]
        e['baseline_'+name+'_replicates_FP32']=value.copy();e['baseline_'+name+'_bits_uint32']=value.view('u4').copy()
    for dest,src in {'actual_delta_u_rad':'actual_delta_u64_rad','actual_halfstep_u_rad':'actual_halfstep64_rad','actual_coordinate_denominator_rad':'actual_denominator64_rad',
        'secant_defined_mask':'secant_defined64','model_pair_numerator_F64':'predicted_numerator64_F64','physical_pair_numerator_F64':'physical_numerator64_F64',
        'model_coordinate_secant_F64':'predicted_secant64_F64','physical_coordinate_secant_F64':'physical_secant64_F64','coordinate_secant_error_F64':'secant_residual64_F64'}.items():e[dest]=pair(src)
    e['secant_undefined_mask']=~e['secant_defined_mask']
    for name,value in [('model',p),('physical',t)]:
        minus=np.empty((8,4,2,58),dtype='f4');plus=np.empty_like(minus)
        for j in range(4):
            for ei in range(2):minus[:,j,ei]=value[:,3+ei*8+j*2];plus[:,j,ei]=value[:,4+ei*8+j*2]
        e[name+'_pair_minus_FP32']=minus;e[name+'_pair_plus_FP32']=plus
        e[name+'_pair_numeric_zero']=e[name+'_pair_numerator_F64']==0
        e[name+'_pair_bitexact_zero']=plus.view('u4')==minus.view('u4')
        e[name+'_secant_finite_mask']=np.isfinite(e[name+'_coordinate_secant_F64'])
    plus=raw['actual_u_plus_rad_FP32'].astype('f8')-um[:,None,None]
    minus=um[:,None,None]-raw['actual_u_minus_rad_FP32'].astype('f8')
    hp=np.stack([plus[:,j,:,q] for j,q in enumerate(math.JOINTS)],1)
    hm=np.stack([minus[:,j,:,q] for j,q in enumerate(math.JOINTS)],1)
    e.update(plus_distance_from_baseline_rad=plus,minus_distance_from_baseline_rad=minus,asymmetric_about_baseline=hp!=hm,
        hplus_coordinate_rad=hp,hminus_coordinate_rad=hm,hplus_minus_hminus_coordinate_rad=hp-hm)
    # Stored association is audited exactly; no requirement that this roundoff is zero.
    e['centered_residual_identity_roundoff_F64']=e['centered_response_error_F64']-((p[:,3:].astype('f8')-t[:,3:].astype('f8'))-(pm-tm)[:,None])
    return result,e


def description(a,block_R=False):
    a=np.asarray(a);n=a.size;axes=tuple(range(a.ndim-1))
    r={'elements':int(n),'component_RMS':None if not n else float(np.sqrt((a*a).mean())),
       'MAE':None if not n else float(abs(a).mean()),'maximum_absolute':None if not n else float(abs(a).max()),
       'per_feature_RMS':[None]*a.shape[-1] if not n else np.sqrt((a*a).mean(axis=axes)).tolist(),
       'per_feature_MAE':[None]*a.shape[-1] if not n else abs(a).mean(axis=axes).tolist(),
       'per_feature_maxabs':[None]*a.shape[-1] if not n else abs(a).max(axis=axes).tolist()}
    if block_R and n:r['raw9_Frobenius_RMS']=float(np.sqrt((a*a).sum(-1).mean()))
    return r


def expected_stats(e,math):
    valid=e['secant_defined_mask'];n=int(valid.sum())
    sources={'constant_baseline_error':e['constant_baseline_error_F64'],'model_centered_response':e['centered_model_response_F64'],
       'physical_centered_response':e['centered_physical_response_F64'],'centered_response_error':e['centered_response_error_F64'],
       'model_coordinate_secant_DEFINED_ONLY':e['model_coordinate_secant_F64'][valid],
       'physical_coordinate_secant_DEFINED_ONLY':e['physical_coordinate_secant_F64'][valid],
       'coordinate_secant_error_DEFINED_ONLY':e['coordinate_secant_error_F64'][valid]}
    stats={}
    for name,(lo,hi,unit,_) in math.BLOCKS.items():
        r={'state_units':unit,'coordinate_secant_units':unit+'/control_rad',**{k:description(v[...,lo:hi],name=='R') for k,v in sources.items()},
           'defined_pair_count':n,'undefined_pairs_kept':64-n,
           'model_encoded_numeric_zero_component_count':int(e['model_pair_numeric_zero'][...,lo:hi].sum()),
           'physical_encoded_numeric_zero_component_count':int(e['physical_pair_numeric_zero'][...,lo:hi].sum()),
           'baseline_physical_range_maximum':float(e['baseline_physical_range_F64'][:,lo:hi].max()),
           'baseline_model_range_maximum':float(e['baseline_model_range_F64'][:,lo:hi].max())}
        stats[name]=r
    feature={'feature_indices':list(range(58)),'state_units':[v[2] for v in math.BLOCKS.values() for _ in range(v[1]-v[0])],
       'defined_pair_count':n,'undefined_pair_count_kept':64-n,'statistics':{}}
    for k,a in sources.items():
        d=description(a);feature['statistics'][k]={'samples_per_feature':int(np.prod(a.shape[:-1])),
           'RMS':d['per_feature_RMS'],'MAE':d['per_feature_MAE'],'maximum_absolute':d['per_feature_maxabs']}
    return stats,feature


def source_join(state,identities,rows,physical,cm,math):
    by_branch={}
    for i,ident in enumerate(identities):
        if type(ident['group_id']) is not int or ident['group_id']!=i:raise ValueError('State452 order')
        bid=ident['branch_id']
        if bid is not None:
            if type(bid) is not int or bid in by_branch or not 0<=bid<152:raise ValueError('Branch bijection')
            by_branch[bid]=i
    if len(identities)!=452 or len(rows)!=152 or set(by_branch)!=set(range(152)):raise ValueError('Complete452/152 required')
    ix=np.asarray([by_branch[i] for i in range(152)],dtype='i8').reshape(8,19)
    pre,p,t,u=[state[k][ix].copy() for k in ('pre_x452_FP32','predicted_x452_FP32','target_x452_FP32','actual_u452_FP32')]
    for a,shape in [(pre,(8,19,58)),(p,(8,19,58)),(t,(8,19,58)),(u,(8,19,20))]:math.f32(a,shape)
    sx,su=physical['snapshot_x'],physical['snapshot_u'];math.f32(sx,(8,58));math.f32(su,(8,20))
    guards={'same_original_snapshot_bits40':pre[...,:40].view('u4')==sx[:,None,:40].view('u4'),
       'same_original_snapshot_numeric18':pre[...,40:]==sx[:,None,40:],
       'internally_same_bits40':pre[...,:40].view('u4')==pre[:,:1,:40].view('u4'),
       'internally_same_numeric18':pre[...,40:]==pre[:,:1,40:],
       'baseline_u_bitexact_original':u[:,:3].view('u4')==su[:,None].view('u4'),
       'all_actual_u_bitexact_branch_authority':u.reshape(152,20).view('u4')==physical['branch_u'].view('u4'),
       'direction_centers_bitexact_original':physical['direction__centers'].view('u4')==su.view('u4')}
    nonj=np.ones((8,16,20),dtype=bool);minus=np.empty((8,4,2,20),dtype='f4');plus=np.empty_like(minus);meta=[];ids=np.empty((8,4,2),dtype='i8')
    for s in range(8):
        path=('A0','A2')[s//4];k=PHASES[s%4]
        for key in ('actuator_ids','joint_ids'):
            v=cm[path][key]
            if len(v)!=20 or len(set(v))!=20 or not all(type(x) is int and x>=0 for x in v):raise ValueError('Native IDs20')
        for j,q in enumerate(math.JOINTS):ids[s,j]=[cm[path]['actuator_ids'][q],cm[path]['joint_ids'][q]]
        for off in range(19):
            bid=s*19+off;base=off<3;j=None if base else (off-3)%8//2;ei=None if base else (off-3)//8
            wanted={'row_id':bid,'snapshot_id':s,'path':path,'k':k,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else math.EPS[ei],
              'sign':0 if base else (-1,1)[(off-3)%2],'baseline_repeat':off if base else None,'joint_index':None if base else math.JOINTS[j],'type':'baseline' if base else 'direction'}
            if not all(type(rows[bid].get(key)) is type(val) and rows[bid][key]==val for key,val in wanted.items()):raise ValueError('Original branch order '+str(bid))
            identity=identities[int(ix[s,off])]
            if any(identity[key]!=val for key,val in [('path',path),('k',k),('branch_id',bid),('mode',wanted['type'])]):raise ValueError('Source452 metadata join')
            ctrl=np.asarray(rows[bid]['actualu']);f32=ctrl.astype('f4')
            if ctrl.shape!=(20,) or not np.array_equal(ctrl.astype('f8'),f32.astype('f8')) or not bits(f32,physical['branch_u'][bid]):raise ValueError('Raw JSON u loss/change')
            meta.append({**wanted,'group_id':int(ix[s,off]),'native_actuator_id':None if base else int(ids[s,j,0]),'native_joint_id':None if base else int(ids[s,j,1])})
            if not base:
                q=math.JOINTS[j];keep=np.arange(20)!=q;nonj[s,off-3,keep]=u[s,off,keep].view('u4')==su[s,keep].view('u4')
        for j in range(4):
            for ei in range(2):minus[s,j,ei]=u[s,3+8*ei+2*j];plus[s,j,ei]=u[s,4+8*ei+2*j]
    delta=plus.astype('f8')-minus.astype('f8');clip=np.empty((8,4,2),dtype=bool)
    for s in range(8):
        for j,q in enumerate(math.JOINTS):
            unit=np.eye(20,dtype='f4')[q]
            for ei,eps in enumerate(math.EPS):clip[s,j,ei]=not(np.array_equal(plus[s,j,ei],su[s]+np.float32(eps)*unit) and np.array_equal(minus[s,j,ei],su[s]-np.float32(eps)*unit))
    guards.update(perturbation_nonj_bitexact_original=nonj,minus_endpoint_bitexact=minus.view('u4')==physical['direction__minus'].view('u4'),
       plus_endpoint_bitexact=plus.view('u4')==physical['direction__plus'].view('u4'),
       legacy_d_eff_numericexact=delta/(2*np.asarray(math.EPS,dtype='f8')[None,None,:,None])==physical['direction__d_eff'],
       center_offset_numericexact=(plus.astype('f8')+minus.astype('f8'))*.5-su[:,None,None].astype('f8')==physical['direction__center_offset'],
       clipping_flag_exact=clip==physical['direction__clipped'],
       source_plus_not_less_than_minus_coordinate=np.stack([delta[:,j,:,q] for j,q in enumerate(math.JOINTS)],1)>=0)
    raw={'pre_visible152_FP32':pre,'predicted_y152_FP32':p,'target_y152_FP32':t,'actual_u152_FP32':u,'branch_group_indices':ix,
       'original_snapshot_visible58_FP32':sx.copy(),'original_snapshot_u_FP32':su.copy(),'actual_u_minus_rad_FP32':minus,'actual_u_plus_rad_FP32':plus,
       'legacy_d_eff_dimensionless':physical['direction__d_eff'].copy(),'center_offset_rad':physical['direction__center_offset'].copy(),
       'legacy_clipped':physical['direction__clipped'].copy(),'native_actuator_joint_ids':ids}
    return raw,guards,meta


def pair_records(raw,e,math):
    records=[]
    for s in range(8):
        for j,q in enumerate(math.JOINTS):
            for ei,eps in enumerate(math.EPS):
                off=3+8*ei+2*j;defined=bool(e['secant_defined_mask'][s,j,ei])
                r={'pair_id':len(records),'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':q,
                   'native_actuator_id':int(raw['native_actuator_joint_ids'][s,j,0]),'native_joint_id':int(raw['native_actuator_joint_ids'][s,j,1]),
                   'epsilon_index':ei,'requested_epsilon_rad':eps,'minus_branch_id':s*19+off,'plus_branch_id':s*19+off+1,
                   'actual_coordinate_denominator_rad':float(e['actual_coordinate_denominator_rad'][s,j,ei]),'secant_defined':defined,
                   'legacy_clipped':bool(raw['legacy_clipped'][s,j,ei]),'asymmetric_about_baseline':bool(e['asymmetric_about_baseline'][s,j,ei]),
                   'status':'DEFINED' if defined else 'UNDEFINED_ZERO_ACTUAL_COORDINATE_DENOMINATOR'}
                for k in ('hplus_coordinate_rad','hminus_coordinate_rad','hplus_minus_hminus_coordinate_rad'):r[k]=float(e[k][s,j,ei])
                records.append(r)
    return records


def audit(config_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__)
    (out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_derived_output_read.json',{'auditor':auditor,'math':pin(MATH),'producer_core_sha':CORE_SHA,'actual_derived_output_read':False,
       'rules':'All152/128/64; explicit(s,j,eps) bijection; F64 cast before subtraction; original numeric clipping; raw q40/base/nonj bits; units separate; no floor/stable/score.'})
    math=module(MATH,MATH_SHA);checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    cfg=read(config_path);report=read(run/'report.json');proto=read(checked(cfg['protocol']));inputs=proto['inputs']
    gate('config_protocol_route_semantic_copies_exact',cfg['schema']=='full151105_saved_action_response_CPU_run:1' and cfg['role']=='saved_fixed_state_offset_vs_finite_action_response'
       and cfg['CPU_run_authorized'] is True and type(cfg['cpu_threads']) is int and cfg['cpu_threads']==1 and cfg['protocol']['sha256']==PROTOCOL_SHA
       and same(report['config'],pin(config_path)) and same(report['protocol'],cfg['protocol']) and strict_json(cfg,read(run/'config.json'))
       and strict_json(proto,read(run/'protocol.json')) and Path(cfg['output']).samefile(run))
    inventory=read(run/'source_inventory.json');tokens=[]
    for p in cfg['source_pins'].values():
        q=checked(p);s=q.stat();tokens.append((s.st_dev,s.st_ino or str(q).casefold()))
    gate('current27_unique_source_closure_and_consumer_v2',len(tokens)==len(set(tokens))==27 and report['source_pins']==inventory
       and len(inventory)==27 and all(any(same(p,q) for q in cfg['source_pins'].values()) for p in inventory.values()) and any(p['sha256']==CORE_SHA for p in inventory.values()))
    for p in [*inventory.values(),*inputs.values(),cfg['protocol'],cfg['cpu_ready']]:checked(p)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zipok=len(z.namelist())==len(inventory) and all(hashlib.sha256(z.read('%04d_%s'%(i,Path(p['path']).name))).hexdigest()==p['sha256'] for i,p in enumerate(inventory.values()))
    gate('all_immutable_source_archive_bytes_current_exact',zipok)
    outputpins=read(run/'output_pins.json')
    for p in outputpins.values():checked(p)
    gate('all_original_immutable_output_pins_exact',True)
    ready=read(checked(cfg['cpu_ready']));proof=read(checked(ready['cpu_report']))
    gate('current_producer42_pure_proof_bound_not_actual_model',proof['passed']==proof['total']==42==len(proof['checks']) and all(v is True for v in proof['checks'].values())
       and ready['source_pins']==proof['source_pins'] and ready['source_pins']['consumer']['sha256']==CORE_SHA and same(ready['protocol'],cfg['protocol'])
       and proof['actual_saved_response_runs']==proof['actual_model_FW_AD_MJB_physics']==proof['actual_saved_array_reads']==0)
    old=read(checked(inputs['state452_saved_independent25']));physicalreview=read(checked(inputs['physical_saved_independent1762']));parent=read(checked(inputs['state452_report']))
    gate('actual_source452_and_physical1762_reviews_bound_separate_scope',old['engineering_qualified'] is True and old['passed']==old['total']==25 and all(v is True for v in old['checks'].values())
       and same(old['config'],inputs['state452_config']) and same(old['producer_report'],inputs['state452_report']) and Path(old['run_dir']).samefile(checked(inputs['state452_report']).parent)
       and physicalreview['passed']==physicalreview['total']==1762 and all(v is True for v in physicalreview['checks'].values())
       and physicalreview['completed_branches']==152 and same(physicalreview['config'],inputs['physical_config']) and same(physicalreview['producer_report'],inputs['physical_report'])
       and physicalreview['physical_gradient_qualified'] is False and physicalreview['whole_hidden_state_qualified'] is False)
    parentpins=read(checked(inputs['state452_report']).parent/'output_pins.json')
    gate('transitive_actual_input_artifact_aliases',all(same(inputs[alias],parentpins[name]) for alias,name in [('state452_arrays','prediction_arrays.npz'),('state452_input_arrays','matched_input_arrays.npz'),('state452_identities','group_identities452.json')])
       and all(same(parent['input_pins'][alias],inputs[alias]) for alias in ('physical_config','physical_arrays','branch_index','snapshot_records')))
    statefull=load(checked(inputs['state452_arrays']));state={k:statefull[k] for k in ('pre_x452_FP32','predicted_x452_FP32','target_x452_FP32','actual_u452_FP32')}
    pp=load(checked(inputs['physical_arrays']));physical={k:pp[k] for k in ('snapshot_x','snapshot_u','branch_u','direction__centers','direction__minus','direction__plus','direction__d_eff','direction__center_offset','direction__clipped')}
    identities=read(checked(inputs['state452_identities']));rows=read(checked(inputs['branch_index']));cm={p:read(checked(inputs[p+'_control_metadata'])) for p in ('A0','A2')}
    raw,guards,metadata=source_join(state,identities,rows,physical,cm,math)
    savedstate=load(run/'selected_source452_raw.npz');savedphys=load(run/'selected_physical_direction_raw.npz');savedraw=load(run/'all152_branch_raw.npz')
    gate('all_parent_selected_source_raw_arrays_exactbytes',set(savedstate)==set(state) and all(bits(v,savedstate[k]) for k,v in state.items()) and set(savedphys)==set(physical) and all(bits(v,savedphys[k]) for k,v in physical.items()))
    gate('all152_raw_branch_and_native_ID_source_join_exactbytes',set(raw)==set(savedraw) and all(bits(v,savedraw[k]) for k,v in raw.items()) and strict_json(metadata,read(run/'all152_branch_metadata.json')))
    sg=load(run/'visible_and_control_guard_masks.npz');summ={k:{'all_passed':bool(v.all()),'failed_elements':int((~v).sum()),'first_failed_indices':np.argwhere(~v)[:20].tolist()} for k,v in guards.items()}
    gate('all_source_guard_masks_and_summary_raw_recomputed',set(sg)==set(guards) and all(bits(v,sg[k]) for k,v in guards.items()) and strict_json(summ,read(run/'guard_summary.json')))
    allpass=all(v.all() for v in guards.values());gate('all_visible58_control_originalsnapshot_guards_passed',allpass)
    independent,e=expected_decomposition(raw,math)
    gate('independent72ad_fixed_state_guard_concordance',independent['fixed_visible_state_interpretation_allowed']==allpass)
    if not allpass or e is None:raise ValueError('BLOCKED_ORIGINAL_GUARD_NO_SUBSET_NO_REPAIR')
    saved=load(run/'decomposition_arrays.npz');mismatch=[k for k,v in e.items() if k not in saved or not equal(v,saved[k])]
    gate('all_F64_response_noise_pair_arrays_masks_and_stored_closure_exact',set(e)==set(saved) and not mismatch)
    details['array_mismatches']=mismatch
    gate('all_F32_replicate_pair_bytes_and_integer_zero_bits_exact',all(bits(v,saved[k]) for k,v in e.items() if v.dtype.kind not in 'fc' or v.dtype==np.float32))
    valid=e['secant_defined_mask'];gate('all64_actual_denominator_secants_undefined_rows_preserved',valid.shape==(8,4,2) and np.isfinite(e['model_coordinate_secant_F64'][valid]).all()
       and np.isnan(e['model_coordinate_secant_F64'][~valid]).all() and np.isnan(e['physical_coordinate_secant_F64'][~valid]).all()
       and report['counts']=={'snapshots':8,'all_branches':152,'baseline_rows':24,'perturbation_endpoints':128,'all_epsilon_pairs':64,'defined_pairs':int(valid.sum()),'undefined_pairs_kept':int((~valid).sum())})
    pair=pair_records(raw,e,math);gate('all64_pair_order_explicit_s_j_epsilon_metadata_and_values',strict_json(pair,read(run/'all64_pair_records.json')))
    stats,features=expected_stats(e,math)
    gate('all_six_original_unit_block_statistics_exact',strict_json(stats,read(run/'statistics.json')) and strict_json(stats,report['statistics']))
    gate('all58_separate_feature_statistics_and_empty_mask_rules_exact',strict_json(features,read(run/'per_feature58_statistics.json')) and strict_json(features,report['per_feature58_statistics']))
    gate('source_recorded_invariants_bound_not_independent_raw_RNG_or_weights',report['invariants']=={'input_files_unchanged':True,'RNG_unchanged':True,'Torch_Warp_MuJoCo_not_imported':True})
    gate('scope_no_new_model_score_floor_stable_or_gradient_qualification',report['schema']=='full151105_saved_action_response_CPU_analysis:1'
       and report['status']=='COMPLETED_SAVED_OFFSET_AND_FINITE_ACTION_RESPONSE_NOT_JACOBIAN' and report['fixed_visible_state_interpretation_permitted'] is True
       and report['model_weight_loads_forward_AD_MJB_Env_GPU_physics']==0 and report['physical_gradient_qualified'] is False and report['whole_hidden_state_qualified'] is False)
    for p in [*inventory.values(),*inputs.values(),*outputpins.values()]:checked(p)
    gate('source_outputs_and_independent_math_unchanged',pin(__file__)==auditor and sha(MATH)==MATH_SHA)
    gate('reviewer_NumPy_only_no_Torch_native_loader_or_physics',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic','scipy')))
    np.savez_compressed(out/'independent_all128_64_response_arrays.npz',**independent['arrays'])
    details.update(pair_order='Independent s,eps,j explicitly mapped by identity to producer s,j,eps; all64 checked.',
       clipped_pairs=int(raw['legacy_clipped'].sum()),asymmetric_about_baseline_pairs=int(e['asymmetric_about_baseline'].sum()),
       minimum_actual_denominator_rad=float(e['actual_coordinate_denominator_rad'].min()),maximum_actual_denominator_rad=float(e['actual_coordinate_denominator_rad'].max()),
       stored_identity_roundoff_maxabs=float(abs(e['centered_residual_identity_roundoff_F64']).max()),
       independent_alternative_identity_association_maxabs=float(abs(independent['arrays']['bias_identity_closure128_F64']).max()),
       model_numeric_zero_components=int(e['model_pair_numeric_zero'].sum()),physical_numeric_zero_components=int(e['physical_pair_numeric_zero'].sum()))
    result={'schema':'full151105_saved_action_response_saved_independent:1','status':'PASS_SAVED_ALL152_RESPONSE_REVIEW_NOT_JACOBIAN' if all(checks.values()) else 'FAILED_SAVED_RESPONSE_REVIEW',
       'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'math':pin(MATH),
       'config':pin(config_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'counts':report['counts'],'statistics':stats,'details':details,
       'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'HO_gradient_qualified':False,'task_quality_qualified':False,
       'limitations':['All8 snapshots/152branches/128endpoints/64dependent pairs retained, no stable/noisefloor or historical9 selection.',
        'Baseline means/ranges/variances are coordinate statistics, not valid poses/actions or noise certificates.',
        'Original FP32 clipping uses numeric equality including signed zeros; q/qd/base/nonjoint byte predicates remain strict.',
        'Secants use actual executed coordinate delta rad; legacy d_eff is dimensionless and halfstep is rad. No second EMA or AD.',
        'Raw F32/ints/bool bytes independently bound; F64 derived values equal including NaN locations, closure roundoff recorded without gate.',
        'Producer RNG/invariant booleans are source-backed; no raw RNG or weights reload and no independent whole hidden state certificate.',
        'Fixed CPU-shadow WM finite response is not historical GPU BF16 replay, true physical Jacobian, HO gradient, or task success.'],
       'new_models_forward_AD_MJB_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'counts':result['counts'],'details':details},indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:raise SystemExit(audit(a.config,a.run_dir,a.output))
    except Exception as e:
        if a.output.exists() and not (a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'original_sources_unchanged':True})
        raise

