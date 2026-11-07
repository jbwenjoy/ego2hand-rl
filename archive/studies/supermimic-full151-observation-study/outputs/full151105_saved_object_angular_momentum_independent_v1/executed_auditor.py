"""Independent all1808/all9473 saved angular arithmetic; no MJB or backend.

Quaternion-vector observer and scalar fsum/cross differ from the producer matrix
chain. F64 comparison tolerances compare implementations, NEVER residual to0.
Actual --run requires a separate Root terminal authorization.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/object_angular_momentum_independent_math_v1.py'
MATH_SHA='9e059f9c677798b4e4026abb0e018f96655e8bbcb8c795fef52e1bca5afdc30b'
CORE=ROOT/'src/full151_gradient/object_angular_momentum105_v1.py'
CORE_SHA='4fb22ef1d940273563b730fc24fb0eb078c07d896f049237b3a17caf3058be67'
PREREG=ROOT/'protocols/full151105_saved_object_angular_momentum_CPU_prereg_v1.json'
PREREG_SHA='a20050a01778d6a55edf90545fe9f0af6032b4dbfde8f705071cd42f953fdca4'
READY=ROOT/'outputs/full151105_object_angular_momentum_preparation_CPU_v1/READY.json'
READY_SHA='ca006af96481b0e2875c7ab5a9439934490d7d5fd488afb478eb4c808639ecd7'
F64_RTOL=2e-12;F64_ATOL=2e-14


def require(v,message):
    if not v:raise ValueError(message)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def checked(v):
    require(type(v) is dict and set(v)=={'path','sha256'} and Path(v['path']).is_absolute(),'Exact absolute pin')
    p=Path(v['path']).resolve(strict=True);require(sha(p)==v['sha256'],'Changed pinned file '+str(p));return p


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def equivalent(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and np.allclose(a,b,rtol=F64_RTOL,atol=F64_ATOL,equal_nan=True)


def math_module():
    require(sha(MATH)==MATH_SHA,'Frozen independent math')
    s=importlib.util.spec_from_file_location('angular_scalar_independent',MATH);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def metadata(protocol_path):
    require(sha(PREREG)==PREREG_SHA and sha(CORE)==CORE_SHA and sha(READY)==READY_SHA,'Frozen observer/source/preparation')
    p,pr=read(protocol_path),read(PREREG)
    require(p['schema']=='full151105_saved_object_angular_momentum_CPU_protocol:1' and p['role']==pr['role'] and p['accuracy_gate'] is None,'Angular descriptive namespace, no accuracy gate')
    require(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False and p['output_prefix']==pr['output_prefix'],'Explicit CPU authorization/fresh scope')
    require(same(p['preregistration'],pin(PREREG)) and set(p['inputs'])==set(pr['inputs']) and len(p['inputs'])==10,'Fixed10 source aliases')
    for k,v in p['inputs'].items():require(same(v,pr['inputs'][k]),'Same original source '+k)
    for v in p['source_pins'].values():checked(v)
    require(any(same(v,pin(CORE)) for v in p['source_pins'].values()),'Current executed observer source bound')
    i=p['inputs'];lr=read(checked(i['linear_report']));meta=read(checked(i['linear_metadata']))
    require(lr['schema']=='full151105_saved_object_linear_momentum_CPU_analysis:1' and lr['status']=='COMPLETED_CONDITIONAL_SAVED_MOMENTUM_NO_ACCURACY_GATE' and lr['all1808_preserved'] is True,'Existing conditional linear metadata authority')
    require(meta==lr['metadata'] and lr['counts']['passive_MJB_returned']==1,'Static metadata inherited from existing once-only read, NOT re-read MJB')
    for k in ('physical_samples','geometry_catalog','actual_context','A0_actual_readbacks','A2_actual_readbacks'):require(same(i[k],lr['input_pins'][k]),'Exact parent data alias '+k)
    inv=read(checked(i['linear_source_inventory']))
    for v in pr['physical_source_pins'].values():require(any(same(v,q) for q in inv.values()),'Original execution source membership')
    ready=read(READY);proof=read(checked(ready['cpu_report']))
    require(proof['passed']==proof['total']==3 and all(v is True for v in proof['checks'].values()) and ready['source_pins']==proof['source_pins'],'Actual focused current-source preparation')
    for v in ready['source_pins'].values():checked(v)
    return p,meta


def field_rebinding(context,readbacks,observed,old,body):
    for field,width in (('body_inertia',3),('body_iquat',4)):
        a=context[field];require(a.dtype==np.float32 and a.ndim==3 and a.shape[0]>0 and a.shape[-1]==width and 0<=body<a.shape[1] and np.isfinite(a).all(),'Actual source full-field ABI')
        require(bits(a,np.ascontiguousarray(np.broadcast_to(a[0],a.shape))),'All source worlds same physical constants')
        require(bits(observed['source_context__'+field],a),'Captured original context full-field bytes')
        for path in ('A0','A2'):
            for stage in ('after_copy','after_set_const_0'):
                key=stage+'__'+field;v=readbacks[path][key]
                require(v.dtype==np.float32 and v.ndim==3 and v.shape[0]>0 and v.shape[1:]==a[0].shape and np.isfinite(v).all(),'Actual readback layout')
                require(bits(v,np.ascontiguousarray(np.broadcast_to(a[0],v.shape))) and bits(v,observed[path+'__'+key]) and bits(v,old[path+'__'+key]),'Each original path/stage full field exact bytes')


def contact_evidence(c,raw,objects,hand,nefc,COM,m):
    pair=np.asarray(c['geom']);require(pair.shape==(2,) and pair.dtype.kind in 'iu' and np.array_equal(pair,np.asarray(raw['geom'])),'All original host/raw ordered geom pairs')
    a,b=map(int,pair);inside=(a in objects,b in objects);external=inside[0]!=inside[1];internal=all(inside)
    typ,world=int(raw['type']),int(raw['worldid']);slots=np.asarray(raw['efc_address']);f=np.asarray(c['force6'],dtype='f8');frame=np.asarray(c['frame'],dtype='f8');pos=np.asarray(c['pos'],dtype='f8')
    reasons=[]
    if slots.ndim!=1 or not slots.size or slots.dtype.kind not in 'iu':reasons.append('RAW_SLOT_ABI')
    else:
        if not (slots>=0).any():reasons.append('NO_ACTIVE_RAW_SLOT')
        if not ((slots==-1)|((slots>=0)&(slots<nefc))).all():reasons.append('RAW_SLOT_RANGE')
    if not typ&1:reasons.append('NO_CONSTRAINT_BIT')
    if world!=0:reasons.append('NONZERO_WORLD')
    if int(c['efc_address'])<0:reasons.append('NEGATIVE_REORDERED_HOST_ADDRESS')
    if not(f.shape==(6,) and frame.shape==(3,3) and np.isfinite(f).all() and np.isfinite(frame).all()):reasons.append('NONFINITE_OR_FORCE_FRAME_ABI')
    if not(pos.shape==COM.shape==(3,) and np.isfinite(pos).all() and np.isfinite(COM).all()):reasons.append('NONFINITE_OR_POS_COM_ABI')
    force=lever=direct=total=np.full(3,np.nan)
    if internal:force=np.zeros(3);lever=np.zeros(3);direct=np.zeros(3);total=np.zeros(3)
    elif external and not reasons:
        v=m.signed_contact_torque(f,frame,pos,COM,1 if inside[1] else -1)
        force=v['force_world_N_F64'];lever=v['lever_torque_COM_Nm_F64'];direct=v['direct_torque_Nm_F64'];total=v['total_torque_COM_Nm_F64']
    category='HO' if external and (a in hand or b in hand) else 'other_external' if external else 'internal' if internal else 'not_object'
    return {'external':external,'internal':internal,'category':category,'constraint_force_evidence_valid':not reasons,'reasons':reasons,
      'signed_force_world_N':force,'lever_torque_world_Nm':lever,'direct_torque_world_Nm':direct,'total_torque_world_Nm':total,
      'raw_force6':f,'raw_type':typ,'raw_worldid':world,'raw_slots':slots.tolist(),'host_efc_address_reordered':int(c['efc_address']),'ordered_geom':pair.tolist(),
      'position_world_saved':pos,'frame_saved':frame,'COM_solver_PRE_saved':COM}


def saved_rotation(q,m):
    raw=np.asarray(q,dtype='f8');require(raw.shape==(4,) and np.isfinite(raw).all(),'Finite saved wxyz')
    norm=math.sqrt(math.fsum(float(x)*float(x) for x in raw))
    if norm==0:return norm,np.full(4,np.nan),np.full((3,3),np.nan)
    unit=raw/norm;R=np.asarray([m.rotate_unit_q(unit,v) for v in ([1,0,0],[0,1,0],[0,0,1])]).T
    return norm,unit,R


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'arithmetic_frozen_before_actual_saved_read.json',{'auditor':auditor,'independent_math':pin(MATH),'actual_angular_values_read':False,'F64_implementation_comparison':{'relative':F64_RTOL,'absolute':F64_ATOL,'purpose':'Compare saved F64 results against alternate arithmetic, NOT compare residual tozero or qualify physics.'}})
    p,meta=metadata(protocol_path);i=p['inputs'];m=math_module();r=read(run/'report.json');sources=read(run/'source_inventory.json');checks={};detail={};numeric_max={}
    def gate(k,v):checks[k]=bool(v)
    def cmp(key,actual,expected):
        aa,ee=np.asarray(actual),np.asarray(expected)
        if aa.shape==ee.shape:
            finite=np.isfinite(aa)&np.isfinite(ee);value=float(np.max(abs(aa[finite]-ee[finite]))) if finite.any() else None
            if value is not None:numeric_max[key]=max(value,numeric_max.get(key) or 0.)
            elif key not in numeric_max:numeric_max[key]=None
        return equivalent(aa,ee)
    gate('authorized_protocol_exact_parent_metadata_and_all10aliases',p['CPU_run_authorized'] is True and same(r['protocol'],pin(protocol_path)) and read(run/'protocol.json')==p and r['input_pins']==i)
    for v in sources.values():checked(v)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:archive=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    outputs=read(run/'output_pins.json')
    for v in outputs.values():checked(v)
    gate('current_original_source_ZIP_and_immutable_outputs_exact',archive and any(same(v,pin(CORE)) for v in sources.values()))
    context=load(checked(i['actual_context']));readbacks={path:load(checked(i[path+'_actual_readbacks'])) for path in ('A0','A2')};old=load(checked(i['linear_metadata_arrays']));obs=load(run/'actual_inertia_observer.npz')
    body,qadr,vadr=meta['object_body_id'],meta['qposadr'],meta['dofadr'];field_rebinding(context,readbacks,obs,old,body)
    I=context['body_inertia'][0,body];qi=context['body_iquat'][0,body];require((I>0).all(),'Positive actual principal inertia')
    norm,unit=m.normalize_q32(qi);require(unit is not None,'Nonzero actual inertial quaternion')
    basis=np.eye(3,dtype='f4');Ibody=np.column_stack([m.endpoint_world_L(np.array([1,0,0,0],dtype='f4'),v,I,qi)['body_L_F64'] for v in basis])
    gate('actual_inertia_iquat_wholefields_bothpaths_bothstages_and_observer',bits(obs['actual_inertia_principal_FP32'],I) and bits(obs['actual_iquat_raw_FP32'],qi) and cmp('inertial_norm',obs['actual_iquat_raw_norm_F64'],norm) and cmp('inertial_unit_q',obs['actual_iquat_normalized_F64'],unit) and cmp('inertia_body',obs['I_body_F64'],Ibody))
    catalog=read(checked(i['geometry_catalog']));objects=set(meta['object_geom_ids']);hand={x['id'] for x in catalog['native_hand_geoms']}
    require(catalog['native_object_body_id']==body and objects=={x['id'] for x in catalog['native_geom_catalog'] if x['native_body_id']==body},'Exact whole object geometry catalog')
    samples=[json.loads(s) for s in checked(i['physical_samples']).read_text().splitlines()];rows=[json.loads(s) for s in (run/'all1808_rows.jsonl').read_text().splitlines()];long=[json.loads(s) for s in (run/'all9473_generated_contacts.jsonl').read_text().splitlines()]
    a=load(run/'all1808_angular_momentum_arrays.npz');ca=load(run/'all9473_contact_torque_arrays.npz')
    require(len(samples)==len(rows)==1808 and len(long)==9473,'All original row/contact population')
    for k,v in a.items():
        require(v.ndim>=1 and v.shape[0]==1808,'Whole saved1808 array '+k)
        if k.endswith('_F64'):require(v.dtype==np.float64,'Declared F64 array '+k)
        elif k.endswith('_FP32'):require(v.dtype==np.float32,'Declared FP32 array '+k)
        else:require(v.dtype==np.bool_ and v.shape==(1808,),'Declared bool mask '+k)
    for k,v in ca.items():
        require(v.shape==((9473,3) if 'torque_world' in k else (9473,)),'Whole saved9473 array '+k)
        require(v.dtype==(np.float64 if 'torque_world' in k else np.int64 if k in ('row_id','contact_index') else np.bool_),'Declared contact storage type '+k)
    dt=np.float32(meta['dt_wp32_s']);clock=[np.float32(0.)]
    for _ in range(600):clock.append(np.float32(clock[-1]+dt))
    static=bool(meta['COM0_isolated_freebody'] and meta['noncontact_metadata_closed'] and not meta['unsupported_noncontact_terms'])
    require(not meta['COM0_isolated_freebody'] or np.array_equal(old['body_ipos'][body],np.zeros(3)),'Inherited COM0 metadata bytes')
    contact_ok=endpoint_ok=primary_ok=identity_ok=True;cursor=0;calculated={};contact_arrays={k:[] for k in ('lever_torque_world_Nm','direct_torque_world_Nm','total_torque_world_Nm','external','internal','evidence_valid','row_id','contact_index')}
    for rid,s in enumerate(samples):
        record=rows[rid];COM=np.asarray(s['object_solver_COM'],dtype='f8');pose=np.asarray(s['object_solver_pose'],dtype='f8');require(COM.shape==(3,) and pose.shape==(7,) and np.isfinite(COM).all() and np.isfinite(pose).all(),'Finite original PRE COM/pose')
        identity={'row_id':rid,**{k:s[k] for k in ('path','mode','branch_id','phase','substep','canonical_pre_tick','canonical_post_tick','pre_time','post_time')}}
        identity_ok &= all(type(record[k]) is type(v) and record[k]==v for k,v in identity.items())
        for key,value in a.items():
            saved=np.asarray(record[key],dtype=value.dtype)
            identity_ok &= saved.shape==value[rid].shape and (np.array_equal(saved,value[rid],equal_nan=True) if value.dtype.kind=='f' else np.array_equal(saved,value[rid]))
        preq,postq=np.asarray(s['pre_qpos'],dtype='f4'),np.asarray(s['post_qpos'],dtype='f4');prev,postv=np.asarray(s['pre_qvel'],dtype='f4'),np.asarray(s['post_qvel'],dtype='f4')
        require(preq.shape==postq.shape==(27,) and prev.shape==postv.shape==(26,) and all(np.isfinite(v).all() for v in (preq,postq,prev,postv)),'Original complete freebody arrays')
        ticks=[s['canonical_pre_tick'],s['canonical_post_tick']];require(all(type(x) is int for x in ticks) and 0<=ticks[0]<ticks[1]<=600 and ticks[1]==ticks[0]+1 and ticks[0]==4*s['phase']+s['substep']-1,'Source adjacent tick phase')
        require(all(float(np.float32(s[k]))==s[k] and np.float32(s[k]).tobytes()==clock[tick].tobytes() for k,tick in zip(('pre_time','post_time'),ticks)),'Original FP32 clock')
        require(type(s['nefc']) is int and s['nefc']>=0 and math.isfinite(float(s['external_force_max'])) and s['external_force_max']>=0,'Original force/count evidence schema')
        pre=m.endpoint_world_L(preq[qadr+3:qadr+7],prev[vadr+3:vadr+6],I,qi);post=m.endpoint_world_L(postq[qadr+3:qadr+7],postv[vadr+3:vadr+6],I,qi)
        for name,qp,qv,v in [('pre',preq,prev,pre),('post',postq,postv,post)]:
            _,_,Rb=saved_rotation(qp[qadr+3:qadr+7],m)
            raw_fields={f'qpos_{name}_FP32':qp,f'qvel_{name}_FP32':qv,f'quat_{name}_raw_FP32':qp[qadr+3:qadr+7],f'omega_body_{name}_FP32':qv[vadr+3:vadr+6]}
            endpoint_ok &= all(bits(a[k][rid],value) for k,value in raw_fields.items())
            expected={f'quat_{name}_raw_norm_F64':v['body_quaternion_norm_F64'],f'quat_{name}_normalized_F64':v['body_unit_quaternion_F64'],f'R_{name}_F64':Rb,
              f'omega_world_{name}_F64':np.full(3,np.nan) if not v['endpoint_valid'] else m.rotate_unit_q(v['body_unit_quaternion_F64'],qv[vadr+3:vadr+6]),f'L_{name}_world_F64':v['world_L_F64']}
            endpoint_ok &= all(cmp(k,a[k][rid],value) for k,value in expected.items()) and bool(a[f'{name}_endpoint_valid'][rid])==v['endpoint_valid']
        sn,solunit,solR=saved_rotation(pose[3:],m);_,_,preR=saved_rotation(preq[qadr+3:qadr+7],m)
        diagnostics={'solver_PRE_pose_saved_F64':pose,'solver_PRE_COM_saved_F64':COM,'solver_PRE_quat_raw_norm_F64':sn,'solver_PRE_quat_normalized_F64':solunit,
          'solver_PRE_minus_pre_qpos_position_F64':pose[:3]-preq[qadr:qadr+3].astype('f8'),'solver_PRE_minus_pre_qpos_R_F64':solR-preR,'solver_PRE_COM_minus_pre_qpos_position_F64':COM-preq[qadr:qadr+3].astype('f8')}
        endpoint_ok &= all(cmp(k,a[k][rid],value) for k,value in diagnostics.items()) and bool(a['solver_PRE_pose_observer_valid'][rid])==(sn>0)
        sums={cat:{term:[] for term in ('lever','direct','total')} for cat in ('HO','other_external')};complete={cat:True for cat in sums};known=[];ext=internal=0
        raw=s['raw_wp_contact'];require(s['nacon']==len(s['contacts']) and all(len(raw[k])==s['nacon'] for k in ('geom','type','worldid','efc_address')),'Original contact counts')
        for ci,c in enumerate(s['contacts']):
            rawrow={k:raw[k][ci] for k in raw};v=contact_evidence(c,rawrow,objects,hand,s['nefc'],COM,m);produced=long[cursor]
            contact_ok &= produced['row_id']==rid and produced['original_contact_index']==ci and all(produced[k]==s[k] for k in ('path','mode','branch_id'))
            for k in ('external','internal','category','constraint_force_evidence_valid','reasons','raw_type','raw_worldid','raw_slots','host_efc_address_reordered','ordered_geom'):contact_ok &= produced[k]==v[k]
            for k in ('raw_force6','position_world_saved','frame_saved','COM_solver_PRE_saved'):contact_ok &= bits(np.asarray(produced[k],dtype='f8'),np.asarray(v[k],dtype='f8'))
            for k in ('signed_force_world_N','lever_torque_world_Nm','direct_torque_world_Nm','total_torque_world_Nm'):contact_ok &= cmp(k,np.asarray(produced[k],dtype='f8'),v[k])
            if v['external']:
                ext+=1
                if v['constraint_force_evidence_valid']:
                    for term,name in [('lever','lever_torque_world_Nm'),('direct','direct_torque_world_Nm'),('total','total_torque_world_Nm')]:sums[v['category']][term].append(v[name])
                    known.append(v['total_torque_world_Nm'])
                else:complete[v['category']]=False
            if v['internal']:internal+=1
            for k,name in [('lever_torque_world_Nm','lever_torque_world_Nm'),('direct_torque_world_Nm','direct_torque_world_Nm'),('total_torque_world_Nm','total_torque_world_Nm')]:contact_arrays[k].append(v[name])
            contact_arrays['external'].append(v['external']);contact_arrays['internal'].append(v['internal']);contact_arrays['evidence_valid'].append(v['constraint_force_evidence_valid']);contact_arrays['row_id'].append(rid);contact_arrays['contact_index'].append(ci);cursor+=1
        totals={cat:{term:m.sum_vectors(values) if complete[cat] else np.full(3,np.nan) for term,values in values_by_term.items()} for cat,values_by_term in sums.items()}
        torque=m.sum_vectors([totals[c]['total'] for c in totals]) if all(complete.values()) else np.full(3,np.nan)
        delta=post['world_L_F64']-pre['world_L_F64'];J=float(dt)*torque;unassigned=delta-J
        eligible=bool(static and all(complete.values()) and pre['endpoint_valid'] and post['endpoint_valid'] and s['external_force_max']==0.)
        residual=unassigned if eligible else np.full(3,np.nan)
        values={'delta_L_world_F64':delta,'tau_total_Nm_F64':torque,'tau_known_valid_contacts_Nm_F64':m.sum_vectors(known),'J_contact_Nms_F64':J,
          'delta_L_minus_contact_J_Nms_F64':unassigned,'closure_residual_Nms_F64':residual,'torque_residual_Nm_F64':residual/float(dt),'known_noncontact_tau_Nm_F64':np.zeros(3) if static else np.full(3,np.nan),
          'original_HO_COM_torque_descriptor_F64':np.asarray(s['hand_object_moment_about_COM_w_Nm'],dtype='f8'),'computed_minus_original_HO_descriptor_F64':totals['HO']['total']-np.asarray(s['hand_object_moment_about_COM_w_Nm'],dtype='f8')}
        for cat,prefix in [('HO','HO'),('other_external','other')]:
            for term in ('lever','direct','total'):values['tau_'+prefix+'_'+term+'_Nm_F64']=totals[cat][term]
        for k,value in values.items():primary_ok &= cmp(k,a[k][rid],value);calculated.setdefault(k,[]).append(value)
        masks={'contact_evidence_complete':all(complete.values()),'HO_complete':complete['HO'],'other_complete':complete['other_external'],'COM0_static_noncontact_closed':static,'closure_interpretation_eligible':eligible}
        primary_ok &= all(a[k].dtype==bool and bool(a[k][rid])==v for k,v in masks.items()) and a['dt_wp32_s_FP32'].dtype==np.float32 and a['dt_wp32_s_FP32'][rid].tobytes()==dt.tobytes() and record['external_object_contacts']==ext and record['internal_object_pairs_canceled']==internal
    identity_counts={'mode':{mode:sum(s['mode']==mode for s in samples) for mode in ('donor','baseline','direction')},'path':{path:sum(s['path']==path for s in samples) for path in ('A0','A2')},'path_mode_phase_substep':{}}
    for s in samples:
        k='/'.join(map(str,(s['path'],s['mode'],s['phase'],s['substep'])));identity_counts['path_mode_phase_substep'][k]=identity_counts['path_mode_phase_substep'].get(k,0)+1
    gate('all1808_original_identities_raw_endpoints_and_alternate_quaternion_vector_L',identity_ok and endpoint_ok and identity_counts==r['identity_counts'])
    gate('all9473_original_contacts_force_validity_and_fsum_lever_direct_once',cursor==9473 and contact_ok and all(equivalent(ca[k],np.asarray(v)) if k not in ('external','internal','evidence_valid','row_id','contact_index') else np.array_equal(ca[k],np.asarray(v)) for k,v in contact_arrays.items()))
    gate('all1808_angular_impulse_residual_masks_and_known_unknown_scope',primary_ok)
    summary=[];res=np.asarray(calculated['torque_residual_Nm_F64'])
    for path in ('ALL','A0','A2'):
        for mode in ('ALL','donor','baseline','direction'):
            ids=[j for j,s in enumerate(samples) if (path=='ALL' or s['path']==path) and (mode=='ALL' or s['mode']==mode)];valid=[j for j in ids if a['closure_interpretation_eligible'][j]];v=res[valid]
            summary.append({'path':path,'mode':mode,'all_rows':len(ids),'eligible_rows':len(valid),'UNKNOWN_rows':len(ids)-len(valid),'eligible_only_torque_residual_component_RMS_Nm':float(np.sqrt(np.mean(v*v))) if valid else None,'eligible_only_torque_residual_vector_RMS_Nm':float(np.sqrt(np.mean(np.sum(v*v,axis=-1)))) if valid else None,'accuracy_gate':None})
    summary_ok=len(r['summary'])==12
    for expected,got in zip(summary,r['summary']):summary_ok &= all(equivalent(got[k],v) if type(v) is float else got[k]==v for k,v in expected.items())
    gate('all12_path_mode_summaries_no_accuracy_threshold_or_row_selection',summary_ok)
    gate('source_counts_zero_backend_MJB_physics_no_physical_qualification',r['schema']=='full151105_saved_object_angular_momentum_CPU_analysis:1' and r['counts']=={'saved_rows_read':1808,'saved_contact_rows':9473,'MJB_Data_backend_model_FK_forward_step_physics_GPU':0} and r['status']=='COMPLETED_CONDITIONAL_ANGULAR_MOMENTUM_NO_ACCURACY_GATE' and r['no_positive_Fn_filter'] is r['no_new_pyramid_decode'] is r['no_gyroscopic_external_world_torque_added'] is True and all(r[k] is False for k in ('physical_force_or_gradient_qualified','full_wrench_qualified','human_force_qualified','whole_hidden_state_qualified')))
    for v in [*sources.values(),*i.values(),*outputs.values()]:checked(v)
    gate('reviewer_no_Torch_native_MJB_models_or_new_physics',not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor and sha(MATH)==MATH_SHA)
    np.savez_compressed(out/'independent_all1808_angular_arithmetic.npz',**{k:np.asarray(v) for k,v in calculated.items()})
    detail.update(F64_implementation_comparison={'relative':F64_RTOL,'absolute':F64_ATOL,'not_a_physical_residual_threshold':True},maximum_implementation_differences=numeric_max,
      metadata_scope='Inherited source/nominal-static ledger from prior one passive metadata read, NOT reviewer MJB/runtime flag observation or Root-crosscheck metadata independence.',observer_scope='Both quaternions explicitly F64 normalized. Independent scalar quaternion-vector/fsum arithmetic, not original GPU bit replay or unique cause identification.',source_files=len(sources))
    result={'schema':'full151105_object_angular_momentum_saved_independent:1','status':'PASS_SAVED_ALL1808_ALL9473_ANGULAR_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_ANGULAR_ARITHMETIC_REVIEW','engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'independent_math':pin(MATH),'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'input_pins':i,'summary':summary,'details':detail,'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'human_force_qualified':False,'reviewer_MJB_backend_models_FW_AD_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');args=ap.parse_args();metadata(args.protocol)
    if not args.run:print(json.dumps({'status':'PASS_SOURCE_JSON_HASH_ONLY_NO_ANGULAR_ARRAY_READ','auditor':pin(__file__),'independent_math':pin(MATH)}));return 0
    require(args.run_dir is not None and args.output is not None,'Fresh actual run/review output required')
    try:return audit(args.protocol,args.run_dir.resolve(strict=True),args.output.resolve())
    except Exception as error:
        if args.output.exists() and not(args.output/'failure.json').exists():dump(args.output/'failure.json',{'error':repr(error),'auditor':pin(__file__),'saved_only':True,'producer_outputs_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
