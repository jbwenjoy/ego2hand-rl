"""REVIEWED_SAVED_DIAGNOSTIC_NOT_ANGULAR_ACCURACY_QUALIFIED.

F64 quaternion normalization is an explicit observer convention, not native
GPU pose replay. Default metadata only; Root alone authorizes --execute.
"""
from __future__ import annotations
import csv,json,math,sys,time,zipfile
from pathlib import Path
from . import object_linear_momentum105_v1 as linear
c=linear.c;ROOT=c.ROOT
read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_saved_object_angular_momentum_CPU_protocol:1'
ROLE='all1808_saved_world_angular_momentum_diagnostic'
PREFIX='full151105_saved_object_angular_momentum_CPU_'
INPUTS={'linear_report','linear_metadata','linear_metadata_arrays','linear_source_inventory','physical_samples','geometry_catalog',
        'actual_context','A0_actual_readbacks','A2_actual_readbacks','source_feasibility'}
OWN={'consumer':ROOT/'src/full151_gradient/object_angular_momentum105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_object_angular_momentum_saved_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_object_angular_momentum_cpu_v1.py'}
BORROWED={'borrowed_linear':ROOT/'src/full151_gradient/object_linear_momentum105_v1.py',
          'borrowed_IO':ROOT/'src/full151_gradient/response_calibration105_v1.py'}
FEASIBILITY=ROOT/'docs/full151_object_angular_momentum_source_feasibility_v1.md'
FEASIBILITY_SHA='f623d08b4ee23e6cca25e48870e01e3f8605c5e6311e198ac4d11866c4e3e658'
PREREG=ROOT/'protocols/full151105_saved_object_angular_momentum_CPU_prereg_v1.json'
PREREG_SHA='a20050a01778d6a55edf90545fe9f0af6032b4dbfde8f705071cd42f953fdca4'


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE,'Angular protocol namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False,'Explicit CPU/noGPU')
    require(p.get('accuracy_gate') is None and p.get('output_prefix')==PREFIX,'No angular accuracy gate/scoped output')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and type(p.get('source_pins')) is dict and p['source_pins'],'Exact10 inputs/source pins')
    if execute:require(p['CPU_run_authorized'],'Root explicit saved angular authorization required')
    return p


def metadata_preflight(protocol_path,output=None):
    require(not any(x in sys.modules for x in ('torch','mujoco','warp')),'Metadata must not initialize backend')
    p=validate_protocol(read(protocol_path));i=p['inputs'];pr=read(PREREG)
    require(same(p['preregistration'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Frozen angular observer/budget parent')
    require(set(i)==set(pr['inputs']) and all(same(v,pr['inputs'][k]) for k,v in i.items()),'Exact scientific input identities')
    for v in i.values():checked(v)
    require(same(i['source_feasibility'],{'path':str(FEASIBILITY),'sha256':FEASIBILITY_SHA}),'Reviewed angular source convention')
    lr=read(checked(i['linear_report']));meta=read(checked(i['linear_metadata']));lp=checked(i['linear_report']).parent
    require(lr.get('schema')=='full151105_saved_object_linear_momentum_CPU_analysis:1' and lr.get('status')=='COMPLETED_CONDITIONAL_SAVED_MOMENTUM_NO_ACCURACY_GATE','Completed conditional linear parent, not force certificate')
    require(lr['counts']['saved_rows_read']==1808 and lr['all1808_preserved'] is True and lr['all_generated_contact_rows']==9473,'Complete original population')
    for alias,name in (('linear_metadata','metadata.json'),('linear_metadata_arrays','passive_metadata_and_actual_readbacks.npz'),('linear_source_inventory','source_inventory.json')):
        require(checked(i[alias]).samefile(lp/name),'Exact saved linear artifact route')
    require(meta==lr['metadata'] and meta['nq']==27 and meta['nv']==26,'Saved passive metadata exact parent')
    for k in ('physical_samples','geometry_catalog','actual_context','A0_actual_readbacks','A2_actual_readbacks'):
        require(same(i[k],lr['input_pins'][k]),'Original data identity: '+k)
    for value in lr['input_pins'].values():checked(value)
    inventory=read(checked(i['linear_source_inventory']))
    require(type(inventory) is dict and inventory,'One saved parent source map')
    for value in pr['physical_source_pins'].values():require(any(same(value,v) for v in inventory.values()),'Original physical source membership')
    sources=c.canonical_sources(list(inventory.values())+list(p['source_pins'].values())+list(i.values())+list(lr['input_pins'].values())+
        [pin(v) for v in list(OWN.values())+list(BORROWED.values())]+[pin(PREREG)])
    for value in list(OWN.values())+list(BORROWED.values()):
        require(any(same(pin(value),v) for v in p['source_pins'].values()),'Current own/helper source must be protocol-bound')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists(),'Fresh output')
    return {'status':'PASS_ANGULAR_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,
            'metadata':meta,'actual_arrays_read_MJB_angular_arithmetic':0}


def declaration():
    return {'schema':'full151105_object_angular_momentum_declaration:1','status':'REVIEWED_SAVED_DIAGNOSTIC_NOT_ANGULAR_ACCURACY_QUALIFIED',
        'protocol_schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),'source_feasibility':pin(FEASIBILITY),
        'sources':{k:pin(v) for k,v in {**OWN,**BORROWED}.items()},'budget':{'saved_rows':1808,'generated_contacts':9473,'MJB_Data_backend_model_physics_GPU':0},
        'actual_arrays_read_MJB_angular_arithmetic':0}


def normalized_rotation(q,np):
    """Chosen F64 observer: preserve q/norm separately; no GPU equivalence claim."""
    raw=np.asarray(q,np.float64);require(raw.shape==(4,) and np.isfinite(raw).all(),'Finite wxyz4')
    norm=float(np.linalg.norm(raw));require(math.isfinite(norm) and norm>0,'Nonzero quaternion observer input')
    normalized=raw/norm;w,x,y,z=normalized
    R=np.array([[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y)],
                [2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x)],
                [2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z]],np.float64)
    return R,norm,normalized


def actual_inertia(context,readbacks,body,np):
    require(set(readbacks)=={'A0','A2'},'Both actual donor paths')
    expected={}
    for field,width in (('body_inertia',3),('body_iquat',4)):
        src=context[field]
        require(src.dtype==np.float32 and src.ndim==3 and src.shape[0]>0 and src.shape[-1]==width and 0<=body<src.shape[1] and np.isfinite(src).all(),'Source actual FP32 '+field)
        first=src[0]
        require(src.tobytes()==np.ascontiguousarray(np.broadcast_to(first,src.shape)).tobytes(),'Fixed source worlds '+field)
        for values in readbacks.values():
            for stage in ('after_copy','after_set_const_0'):
                got=values[stage+'__'+field]
                require(got.dtype==np.float32 and got.ndim==3 and got.shape[0]>0 and got.shape[1:]==first.shape and np.isfinite(got).all(),'Actual nonempty field '+field)
                require(got.tobytes()==np.ascontiguousarray(np.broadcast_to(first,got.shape)).tobytes(),'Both paths/both stages exact actual field '+field)
        expected[field]=first[body].copy()
    principal=expected['body_inertia'].astype(np.float64);require((principal>0).all(),'Positive actual principal inertia')
    Q,norm,iq=normalized_rotation(expected['body_iquat'],np);I=Q@np.diag(principal)@Q.T
    return I,{'actual_inertia_principal_FP32':expected['body_inertia'],'actual_iquat_raw_FP32':expected['body_iquat'],
        'actual_iquat_raw_norm_F64':np.asarray(norm),'actual_iquat_normalized_F64':iq,'I_body_F64':I}


def endpoint(qpos,qvel,qadr,vadr,I,np):
    qp=np.asarray(qpos,np.float32);qv=np.asarray(qvel,np.float32)
    require(qp.shape==(27,) and qv.shape==(26,) and np.isfinite(qp).all() and np.isfinite(qv).all(),'Original full27/full26 finite')
    require(type(qadr) is int and type(vadr) is int and 0<=qadr<=20 and 0<=vadr<=20,'Bound freejoint source addresses')
    raw=qp[qadr+3:qadr+7].copy();omega=qv[vadr+3:vadr+6].copy();norm=float(np.linalg.norm(raw.astype(np.float64)))
    valid=bool(math.isfinite(norm) and norm>0)
    if valid:R,norm,normalized=normalized_rotation(raw,np)
    else:R=np.full((3,3),np.nan);normalized=np.full(4,np.nan)
    L=R@(I@omega.astype(np.float64))
    if valid:require(np.isfinite(L).all(),'Finite angular momentum observer')
    return {'qpos':qp,'qvel':qv,'quat_raw':raw,'quat_raw_norm':norm,'quat_normalized':normalized,'R':R,
        'omega_body':omega,'omega_world':R@omega.astype(np.float64),'L_world':L,'valid':valid}


def contact_torque(contact,raw,object_geoms,hand,nefc,COM,np):
    q=linear.contact_force(contact,raw,object_geoms,hand,nefc,np)
    pos=np.asarray(contact['pos'],np.float64);COM=np.asarray(COM,np.float64);frame=np.asarray(contact['frame'],np.float64)
    position_valid=pos.shape==COM.shape==(3,) and np.isfinite(pos).all() and np.isfinite(COM).all()
    if not position_valid:q['constraint_force_evidence_valid']=False;q['reasons'].append('NONFINITE_OR_POS_COM_ABI')
    lever=direct=total=np.full(3,np.nan,np.float64)
    if q['internal']:lever=np.zeros(3);direct=np.zeros(3);total=np.zeros(3)
    elif q['external'] and q['constraint_force_evidence_valid']:
        sign=1. if q['ordered_geom'][1] in object_geoms else -1.
        lever=np.cross(pos-COM,q['signed_force_world_N']);direct=sign*(frame.T@q['raw_force6'][3:]);total=lever+direct
    q.update(position_world_saved=pos,frame_saved=frame,COM_solver_PRE_saved=COM,lever_torque_world_Nm=lever,
        direct_torque_world_Nm=direct,total_torque_world_Nm=total,direct_torque_included_once=True)
    return q


def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];meta=m['metadata'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    counts={'saved_rows_read':0,'saved_contact_rows':0,'MJB_Data_backend_model_FK_forward_step_physics_GPU':0};phase='actual_inertia_rebinding'
    try:
        import numpy as np
        def npz(value):
            with np.load(checked(value),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
        context=npz(i['actual_context']);readbacks={path:npz(i[path+'_actual_readbacks']) for path in ('A0','A2')}
        old=npz(i['linear_metadata_arrays']);body=meta['object_body_id'];I,inertia=actual_inertia(context,readbacks,body,np)
        require(not meta['COM0_isolated_freebody'] or np.array_equal(old['body_ipos'][body],np.zeros(3)),'Saved zero COM metadata')
        for path,values in readbacks.items():
            for field in ('body_inertia','body_iquat'):
                for stage in ('after_copy','after_set_const_0'):
                    key=stage+'__'+field;require(old[path+'__'+key].dtype==values[key].dtype and old[path+'__'+key].shape==values[key].shape and old[path+'__'+key].tobytes()==values[key].tobytes(),'Original/saved readback bytes')
        observed={**inertia,**{'source_context__'+k:context[k] for k in ('body_inertia','body_iquat')}}
        observed.update({path+'__'+key:value for path,values in readbacks.items() for key,value in values.items() if key.endswith('__body_inertia') or key.endswith('__body_iquat')})
        np.savez_compressed(out/'actual_inertia_observer.npz',**observed)
        catalog=read(checked(i['geometry_catalog']));objects=set(meta['object_geom_ids']);hand={v['id'] for v in catalog['native_hand_geoms']}
        require(catalog['native_object_body_id']==body and catalog['native_ngeom']==meta['ngeom'] and objects=={v['id'] for v in catalog['native_geom_catalog'] if v['native_body_id']==body},'Original whole object catalog')
        dt=np.float32(meta['dt_wp32_s']);require(np.isfinite(dt) and dt>0,'Original source timestep');clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+dt))
        phase='all_original_1808_rows';samples=[json.loads(s) for s in checked(i['physical_samples']).read_text(encoding='utf-8').splitlines()]
        counts['saved_rows_read']=len(samples);require(len(samples)==1808,'All1808 original rows')
        identity={'mode':{q:sum(s['mode']==q for s in samples) for q in ('donor','baseline','direction')},'path':{q:sum(s['path']==q for s in samples) for q in ('A0','A2')}}
        require(identity['mode']=={'donor':1200,'baseline':96,'direction':512} and identity['path']=={'A0':904,'A2':904},'Original whole population')
        identity['path_mode_phase_substep']={};rows=[];contacts=[]
        qadr,vadr=meta['qposadr'],meta['dofadr'];static=bool(meta['COM0_isolated_freebody'] and meta['noncontact_metadata_closed'] and not meta['unsupported_noncontact_terms'])
        for row_id,s in enumerate(samples):
            linear.saved_velocity_clock(s,meta['nv'],vadr,clock,np)
            key='/'.join(map(str,(s['path'],s['mode'],s['phase'],s['substep'])));identity['path_mode_phase_substep'][key]=identity['path_mode_phase_substep'].get(key,0)+1
            pre=endpoint(s['pre_qpos'],s['pre_qvel'],qadr,vadr,I,np);post=endpoint(s['post_qpos'],s['post_qvel'],qadr,vadr,I,np)
            COM=np.asarray(s['object_solver_COM'],np.float64);pose=np.asarray(s['object_solver_pose'],np.float64)
            require(COM.shape==(3,) and pose.shape==(7,) and np.isfinite(COM).all() and np.isfinite(pose).all(),'Finite original solverPRE pose/COM')
            normsolver=float(np.linalg.norm(pose[3:]));solver_valid=bool(math.isfinite(normsolver) and normsolver>0)
            if solver_valid:Rsolver,normsolver,normalizedsolver=normalized_rotation(pose[3:],np)
            else:Rsolver=np.full((3,3),np.nan);normalizedsolver=np.full(4,np.nan)
            raw=s['raw_wp_contact'];require(type(s['nacon']) is int and type(s['nefc']) is int and s['nacon']>=0 and s['nefc']>=0 and s['nacon']==len(s['contacts']),'Original counts')
            require(all(len(raw[k])==s['nacon'] for k in ('geom','type','worldid','efc_address')),'Raw/host whole contact columns')
            sums={cat:{term:np.zeros(3) for term in ('lever','direct','sum')} for cat in ('HO','other_external')};complete={cat:True for cat in sums};known=np.zeros(3);internal=external=0
            for ci,contact in enumerate(s['contacts']):
                q=contact_torque(contact,{k:raw[k][ci] for k in ('geom','type','worldid','efc_address')},objects,hand,s['nefc'],COM,np)
                if q['external']:
                    external+=1
                    if q['constraint_force_evidence_valid']:
                        for term,name in (('lever','lever_torque_world_Nm'),('direct','direct_torque_world_Nm'),('sum','total_torque_world_Nm')):sums[q['category']][term]+=q[name]
                        known+=q['total_torque_world_Nm']
                    else:complete[q['category']]=False
                if q['internal']:internal+=1
                contacts.append({'row_id':row_id,'original_contact_index':ci,'path':s['path'],'mode':s['mode'],'branch_id':s['branch_id'],**q})
            force_complete=all(complete.values())
            for cat in sums:
                if not complete[cat]:sums[cat]={term:np.full(3,np.nan) for term in ('lever','direct','sum')}
            torque=sums['HO']['sum']+sums['other_external']['sum'];delta=post['L_world']-pre['L_world'];J=float(dt)*torque;unassigned=delta-J
            eligible=bool(static and force_complete and pre['valid'] and post['valid'] and s['external_force_max']==0.)
            residual=unassigned if eligible else np.full(3,np.nan)
            originalHO=np.asarray(s['hand_object_moment_about_COM_w_Nm'],np.float64);require(originalHO.shape==(3,) and np.isfinite(originalHO).all(),'Original HO descriptor')
            record={'row_id':row_id,'path':s['path'],'mode':s['mode'],'branch_id':s['branch_id'],'phase':s['phase'],'substep':s['substep'],
                'canonical_pre_tick':s['canonical_pre_tick'],'canonical_post_tick':s['canonical_post_tick'],'pre_time':s['pre_time'],'post_time':s['post_time'],
                'qpos_pre_FP32':pre['qpos'],'qpos_post_FP32':post['qpos'],'qvel_pre_FP32':pre['qvel'],'qvel_post_FP32':post['qvel'],
                'quat_pre_raw_FP32':pre['quat_raw'],'quat_post_raw_FP32':post['quat_raw'],'quat_pre_raw_norm_F64':pre['quat_raw_norm'],'quat_post_raw_norm_F64':post['quat_raw_norm'],
                'quat_pre_normalized_F64':pre['quat_normalized'],'quat_post_normalized_F64':post['quat_normalized'],'R_pre_F64':pre['R'],'R_post_F64':post['R'],
                'omega_body_pre_FP32':pre['omega_body'],'omega_body_post_FP32':post['omega_body'],'omega_world_pre_F64':pre['omega_world'],'omega_world_post_F64':post['omega_world'],
                'L_pre_world_F64':pre['L_world'],'L_post_world_F64':post['L_world'],'delta_L_world_F64':delta,'dt_wp32_s_FP32':dt,
                'solver_PRE_pose_saved_F64':pose,'solver_PRE_COM_saved_F64':COM,'solver_PRE_quat_raw_norm_F64':normsolver,'solver_PRE_quat_normalized_F64':normalizedsolver,
                'solver_PRE_minus_pre_qpos_position_F64':pose[:3]-pre['qpos'][qadr:qadr+3].astype(np.float64),'solver_PRE_minus_pre_qpos_R_F64':Rsolver-pre['R'],
                'solver_PRE_COM_minus_pre_qpos_position_F64':COM-pre['qpos'][qadr:qadr+3].astype(np.float64),
                'tau_HO_lever_Nm_F64':sums['HO']['lever'],'tau_HO_direct_Nm_F64':sums['HO']['direct'],'tau_HO_total_Nm_F64':sums['HO']['sum'],
                'tau_other_lever_Nm_F64':sums['other_external']['lever'],'tau_other_direct_Nm_F64':sums['other_external']['direct'],'tau_other_total_Nm_F64':sums['other_external']['sum'],
                'tau_total_Nm_F64':torque,'tau_known_valid_contacts_Nm_F64':known,'J_contact_Nms_F64':J,'delta_L_minus_contact_J_Nms_F64':unassigned,
                'closure_residual_Nms_F64':residual,'torque_residual_Nm_F64':residual/float(dt),'known_noncontact_tau_Nm_F64':np.zeros(3) if static else np.full(3,np.nan),
                'original_HO_COM_torque_descriptor_F64':originalHO,'computed_minus_original_HO_descriptor_F64':sums['HO']['sum']-originalHO,
                'contact_evidence_complete':force_complete,'HO_complete':complete['HO'],'other_complete':complete['other_external'],
                'pre_endpoint_valid':pre['valid'],'post_endpoint_valid':post['valid'],'solver_PRE_pose_observer_valid':solver_valid,
                'COM0_static_noncontact_closed':static,'closure_interpretation_eligible':eligible,'external_object_contacts':external,'internal_object_pairs_canceled':internal}
            rows.append(record)
        counts['saved_contact_rows']=len(contacts);require(len(contacts)==9473,'All9473 original generated contacts, not all object forces')
        keys=[k for k,v in rows[0].items() if isinstance(v,(np.ndarray,np.float32)) or k.endswith('_F64') or type(v) is bool]
        arrays={k:np.asarray([r[k] for r in rows]) for k in keys};np.savez_compressed(out/'all1808_angular_momentum_arrays.npz',**arrays)
        np.savez_compressed(out/'all9473_contact_torque_arrays.npz',row_id=np.asarray([q['row_id'] for q in contacts],np.int64),contact_index=np.asarray([q['original_contact_index'] for q in contacts],np.int64),
            lever_torque_world_Nm=np.asarray([q['lever_torque_world_Nm'] for q in contacts]),direct_torque_world_Nm=np.asarray([q['direct_torque_world_Nm'] for q in contacts]),
            total_torque_world_Nm=np.asarray([q['total_torque_world_Nm'] for q in contacts]),external=np.asarray([q['external'] for q in contacts],bool),internal=np.asarray([q['internal'] for q in contacts],bool),
            evidence_valid=np.asarray([q['constraint_force_evidence_valid'] for q in contacts],bool))
        for name,values in (('all1808_rows',rows),('all9473_generated_contacts',contacts)):
            with (out/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for value in values:f.write(json.dumps(linear.jsonable(value),allow_nan=False)+'\n')
            with (out/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader()
                for value in values:writer.writerow({k:json.dumps(linear.jsonable(v)) if isinstance(v,(list,dict)) or hasattr(v,'tolist') else linear.jsonable(v) for k,v in value.items()})
        summary=[]
        for path in ('ALL','A0','A2'):
            for mode in ('ALL','donor','baseline','direction'):
                ids=[j for j,s in enumerate(samples) if (path=='ALL' or s['path']==path) and (mode=='ALL' or s['mode']==mode)]
                eligible=[j for j in ids if rows[j]['closure_interpretation_eligible']];a=arrays['torque_residual_Nm_F64'][eligible]
                summary.append({'path':path,'mode':mode,'all_rows':len(ids),'eligible_rows':len(eligible),'UNKNOWN_rows':len(ids)-len(eligible),
                    'eligible_only_torque_residual_component_RMS_Nm':float(np.sqrt(np.mean(a*a))) if eligible else None,
                    'eligible_only_torque_residual_vector_RMS_Nm':float(np.sqrt(np.mean(np.sum(a*a,axis=-1)))) if eligible else None,'accuracy_gate':None})
        for value in m['sources'].values():checked(value)
        for value in i.values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Protocol unchanged')
        report={'schema':'full151105_saved_object_angular_momentum_CPU_analysis:1','status':'COMPLETED_CONDITIONAL_ANGULAR_MOMENTUM_NO_ACCURACY_GATE',
            'protocol':m['protocol_pin'],'input_pins':i,'counts':counts,'identity_counts':identity,'summary':summary,'all1808_and9473_preserved':True,
            'observer_convention':'F64 normalization of both actual FP32 body_iquat and raw endpoint wxyz; raw/norms preserved; not native GPU bitexact pose replay',
            'known_gravity_COM_torque':'zero only under COM0/static noncontact source conditions','no_gyroscopic_external_world_torque_added':True,
            'conditional_omissions':'effective qacc/solver gradient/generalized passive/constraint terms absent; implicit gyro/orientation/time discretization and observer arithmetic not uniquely separated',
            'HO_original_aggregate_comparison':'descriptor only, no numerical pass threshold','no_positive_Fn_filter':True,'no_new_pyramid_decode':True,
            'physical_force_or_gradient_qualified':False,'full_wrench_qualified':False,'human_force_qualified':False,'whole_hidden_state_qualified':False,
            'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:c._write_json(out/'failure.json',{'schema':'full151105_angular_momentum_failure:1','phase':phase,'error':repr(error),'counts':counts,'no_automatic_retry':True})
        except BaseException:pass
        raise
