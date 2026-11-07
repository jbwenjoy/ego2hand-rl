"""REVIEWED_SAVED_DIAGNOSTIC_NOT_ACCURACY_QUALIFIED: conditional COM momentum.

Default JSON/hash only. --execute permits ONE passive nominal MJB constants read,
never MjData/FK/forward/step/Torch/GPU. Decoded saved contact6 is not re-decoded.
Unknown force evidence/noncontact terms remain UNKNOWN, never silently zero.
"""
from __future__ import annotations
import csv,json,math,os,sys,time,zipfile
from pathlib import Path
from . import response_calibration105_v1 as c
ROOT=c.ROOT
SCHEMA='full151105_saved_object_linear_momentum_CPU_protocol:1'
ROLE='all1808_saved_linear_object_momentum_diagnostic'
PREREG=ROOT/'protocols/full151105_saved_object_linear_momentum_CPU_prereg_v1.json'
PREREG_SHA='4c872bfbf7d3a6e58b10c118155928b4f7e06221a03a062591524632d89b8767'
PREFIX='full151105_saved_object_linear_momentum_CPU_'
INPUTS={'physical_config','physical_report','physical_samples','physical_independent1762','physical_source_inventory','geometry_catalog',
        'nominal_model','actual_context','A0_actual_readbacks','A2_actual_readbacks','source_feasibility','source_contract','source_root_binding'}
OWN={'consumer':ROOT/'src/full151_gradient/object_linear_momentum105_v1.py','CLI':ROOT/'scripts/diagnose_full151_object_linear_momentum_saved_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_object_linear_momentum_cpu_v1.py'}
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE,'New momentum protocol')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False,'Explicit CPU/noGPU')
    require(p.get('accuracy_gate') is None and p.get('output_prefix')==PREFIX,'No accuracy gate/scoped output')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and p.get('source_pins'),'Exact13 input pins and own sources')
    if execute:require(p['CPU_run_authorized'],'Root explicit saved momentum authorization required')
    return p


def metadata_preflight(protocol_path,output=None):
    require(not any(x in sys.modules for x in ('torch','mujoco','warp')),'Fresh metadata path')
    p=validate_protocol(read(protocol_path));i=p['inputs'];pre=read(PREREG)
    require(same(p['preregistration'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Frozen scientific parent')
    require(set(i)==set(pre['inputs']) and all(same(v,pre['inputs'][k]) for k,v in i.items()),'Exact scientific inputs')
    r=read(checked(i['physical_report']));cfg=read(checked(i['physical_config']));review=read(checked(i['physical_independent1762']));c.allchecks(review)
    require(r['status']=='COMPLETED_PHYSICAL_PROBE' and r['episode_step_returned']==1808 and r['baseline_all_passed'] is True,'Original complete physical run')
    require(review['engineering_qualified'] is True and same(review['config'],i['physical_config']) and same(review['producer_report'],i['physical_report']),'Original engineering provenance only')
    inv=read(checked(i['physical_source_inventory']));require(inv==cfg['source_pins'],'Actual original source map')
    for v in pre['physical_source_pins'].values():require(any(same(v,x) for x in inv.values()),'Source review not original execution source')
    root=read(checked(i['source_root_binding']));c.allchecks(root)
    bundle=read(checked(cfg['input_bundle']))
    require(same(bundle['model'],i['nominal_model']) and same(bundle['context'],i['actual_context']),'Model/context actual identity')
    for path in ('A0','A2'):
        require(checked(i[path+'_actual_readbacks']).samefile(checked(i['physical_report']).parent/('donor_'+path)/'actual_context_readbacks.npz'),'Actual readback route')
    sources=c.canonical_sources(list(p['source_pins'].values())+list(inv.values())+list(i.values())+list(pre['physical_source_pins'].values())+[pin(PREREG)]+[pin(x) for x in OWN.values()])
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists(),'Fresh output')
    return {'status':'PASS_MOMENTUM_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,'actual_rows_read_MJB_arithmetic':0}


def declaration():
    return {'schema':'full151105_object_linear_momentum_declaration:1','status':'REVIEWED_SAVED_DIAGNOSTIC_NOT_ACCURACY_QUALIFIED','protocol_schema':SCHEMA,
        'role':ROLE,'input_aliases':sorted(INPUTS),'preregistration':pin(PREREG),'budget':{'passive_MJB':1,'saved_rows':1808,'MjData_FK_forward_step_Torch_GPU':0},
        'sources':{k:pin(v) for k,v in OWN.items()},'actual_rows_read_MJB_arithmetic':0}


def inspect_constants(model,mujoco,np):
    """No mjData or dynamic query. Missing/unsupported terms block closure."""
    arrays={};missing=[]
    names=('body_parentid','body_rootid','body_weldid','body_jntadr','body_jntnum','body_ipos','body_mass','body_gravcomp',
        'jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','jnt_limited','jnt_stiffness','jnt_stiffnesspoly','jnt_actgravcomp',
        'dof_armature','dof_damping','dof_dampingpoly','dof_frictionloss','geom_bodyid','geom_contype','geom_conaffinity','geom_fluid',
        'eq_type','eq_objtype','eq_obj1id','eq_obj2id','actuator_trntype','actuator_trnid','site_bodyid',
        'tendon_stiffness','tendon_stiffnesspoly','tendon_damping','tendon_dampingpoly','tendon_frictionloss')
    for name in names:
        if hasattr(model,name):arrays[name]=np.asarray(getattr(model,name)).copy()
        else:missing.append(name)
    body=int(model.body('object/cube').id);joint=int(model.joint('object/cube_freejoint').id)
    nq,nv=int(model.nq),int(model.nv);qadr=int(model.jnt_qposadr[joint]);vadr=int(model.jnt_dofadr[joint])
    direct_geoms=np.flatnonzero(np.asarray(model.geom_bodyid)==body).tolist()
    descendants=[]
    for b in range(1,int(model.nbody)):
        t=b
        while t>0 and t!=body:t=int(model.body_parentid[t])
        if t==body and b!=body:descendants.append(b)
    structure=bool(all(name in arrays for name in ('body_parentid','body_rootid','body_weldid','body_jntnum','body_ipos','jnt_type','jnt_bodyid'))
        and int(model.jnt_type[joint])==int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[joint])==body
        and int(model.body_parentid[body])==0 and int(model.body_jntnum[body])==1 and not descendants
        and np.array_equal(np.asarray(model.body_ipos[body]),np.zeros(3)) and int(model.body_rootid[body])==body and int(model.body_weldid[body])==body)
    unsupported=list('MISSING_'+x for x in missing)
    def zero(field,where):
        if field in arrays and np.any(arrays[field][where]!=0):unsupported.append('NONZERO_'+field)
    for field in ('dof_armature','dof_damping','dof_dampingpoly','dof_frictionloss'):zero(field,slice(vadr,vadr+6))
    for field in ('jnt_stiffness','jnt_stiffnesspoly','jnt_actgravcomp','jnt_limited'):zero(field,joint)
    zero('body_gravcomp',body)
    # Tendon coupling cannot be recovered from unrecorded dynamic moment arms.
    if int(model.ntendon)>0:unsupported.append('TENDON_COUPLING_NOT_ESTABLISHED')
    if not hasattr(model,'nplugin') or int(model.nplugin)!=0:unsupported.append('PLUGIN_ABSENT_OR_UNSUPPORTED')
    for j in range(int(model.neq)):
        if 'eq_objtype' not in arrays:break
        typ=int(arrays['eq_objtype'][j]);ids=[int(arrays[k][j]) for k in ('eq_obj1id','eq_obj2id')]
        if typ==int(mujoco.mjtObj.mjOBJ_BODY) and body in ids:unsupported.append('OBJECT_BODY_EQUALITY')
        elif typ==int(mujoco.mjtObj.mjOBJ_JOINT) and joint in ids:unsupported.append('OBJECT_JOINT_EQUALITY')
        elif typ not in (int(mujoco.mjtObj.mjOBJ_BODY),int(mujoco.mjtObj.mjOBJ_JOINT)):unsupported.append('UNSUPPORTED_EQUALITY_COUPLING_TYPE')
    for j in range(int(model.nu)):
        typ=int(model.actuator_trntype[j]);ids=np.asarray(model.actuator_trnid[j]).tolist()
        if typ in (int(mujoco.mjtTrn.mjTRN_JOINT),int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            if ids[0]==joint:unsupported.append('OBJECT_ACTUATOR_JOINT')
        elif typ==int(mujoco.mjtTrn.mjTRN_BODY):
            if ids[0]==body:unsupported.append('OBJECT_ACTUATOR_BODY')
        elif typ==int(mujoco.mjtTrn.mjTRN_SITE):
            if any(s>=0 and int(model.site_bodyid[s])==body for s in ids):unsupported.append('OBJECT_ACTUATOR_SITE')
        else:unsupported.append('UNSUPPORTED_ACTUATOR_TRANSMISSION')
    opts={}
    for name in ('timestep','gravity','density','viscosity','wind','integrator','solver','iterations','tolerance','ls_iterations','ls_tolerance','disableflags','enableflags'):
        if hasattr(model.opt,name):
            value=np.asarray(getattr(model.opt,name));opts[name]=value.tolist();arrays['opt_'+name]=value.copy()
        else:unsupported.append('MISSING_OPTION_'+name)
    for name in ('density','viscosity','wind'):
        if name in opts and np.any(np.asarray(opts[name])!=0):unsupported.append('NONZERO_FLUID_'+name)
    if int(model.opt.enableflags)&int(mujoco.mjtEnableBit.mjENBL_SLEEP):unsupported.append('SLEEP_ENABLED')
    if int(model.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):unsupported.append('UNREVIEWED_INTEGRATOR')
    dt=np.float32(model.opt.timestep);gravity_disabled=bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
    arrays['source_effective_WP_gravity_FP32']=np.asarray(model.opt.gravity,np.float32).copy()
    arrays['source_effective_WP_timestep_FP32']=np.asarray(dt,np.float32)
    require(float(dt)>0 and np.isfinite(dt) and np.isfinite(arrays['source_effective_WP_gravity_FP32']).all(),'Finite positive source options')
    metadata={'object_body_id':body,'object_joint_id':joint,'qposadr':qadr,'dofadr':vadr,'nq':nq,'nv':nv,'nbody':int(model.nbody),'ngeom':int(model.ngeom),
        'object_geom_ids':direct_geoms,'object_descendants':descendants,'COM0_isolated_freebody':structure,'unsupported_noncontact_terms':sorted(set(unsupported)),
        'noncontact_metadata_closed':structure and not unsupported,'nominal_options':opts,'dt_wp32_s':float(dt),'dt_source':'F32(nominal opt.timestep); source-bound WP conversion/FP32 clock, not time difference',
        'nominal_tolerance':float(model.opt.tolerance),'effective_WP_tolerance_source_floor':max(float(model.opt.tolerance),1e-6),
        'effective_WP_tolerance_source_FP32':float(np.float32(max(float(model.opt.tolerance),1e-6))),
        'gravity_disabled':gravity_disabled,'gravity_effective_world_m_s2':np.zeros(3).tolist() if gravity_disabled else np.asarray(model.opt.gravity,dtype=np.float32).astype(np.float64).tolist(),
        'options_actual_runtime_readback':False,'WP_callback_evidence':'source-pinned io creates new types.Callback(); no CPU callback getter equated to WP state',
        'applied_forces_evidence':'source-backed worker raw() zero assertions + saved external_force_max; raw applied vectors not independently serialized',
        'solver_effective_qacc_grad_passive_constraint_terms':'NOT_SAVED; finite solver/implicit integration residual remains unidentifiable'}
    return metadata,arrays


def actual_mass(context,readbacks,body,np):
    expected=context['body_mass'][0];require(expected.dtype==np.float32 and np.isfinite(expected).all(),'Source actual FP32 mass field')
    for path,values in readbacks.items():
        for stage in ('after_copy','after_set_const_0'):
            got=values[stage+'__body_mass'];target=np.broadcast_to(expected,got.shape)
            require(got.dtype==np.float32 and got.tobytes()==np.ascontiguousarray(target).tobytes(),'Both paths/both stages actual mass must equal source')
    mass=float(expected[body]);require(math.isfinite(mass) and mass>0,'Positive actual WP mass, never nominal replacement');return mass


def contact_force(contact,raw,object_geoms,hand,nefc,np):
    """Saved signed local force only. Unknown evidence never becomes zero."""
    pair=np.asarray(contact['geom']);require(pair.shape==(2,) and pair.dtype.kind in 'iu','Ordered geom2')
    require(np.array_equal(pair,np.asarray(raw['geom'])),'Raw/host contact order')
    g1,g2=map(int,pair);inside=(g1 in object_geoms,g2 in object_geoms);external=inside[0]!=inside[1];internal=all(inside)
    typ=int(raw['type']);world=int(raw['worldid']);slots=np.asarray(raw['efc_address']);force=np.asarray(contact['force6'],dtype=np.float64);frame=np.asarray(contact['frame'],dtype=np.float64)
    reasons=[]
    if slots.ndim!=1 or not slots.size or slots.dtype.kind not in 'iu':reasons.append('RAW_SLOT_ABI')
    else:
        if not (slots>=0).any():reasons.append('NO_ACTIVE_RAW_SLOT')
        if not ((slots==-1)|((slots>=0)&(slots<nefc))).all():reasons.append('RAW_SLOT_RANGE')
    if not typ&1:reasons.append('NO_CONSTRAINT_BIT')
    if world!=0:reasons.append('NONZERO_WORLD')
    if int(contact['efc_address'])<0:reasons.append('NEGATIVE_REORDERED_HOST_ADDRESS')
    finite=force.shape==(6,) and frame.shape==(3,3) and np.isfinite(force).all() and np.isfinite(frame).all()
    if not finite:reasons.append('NONFINITE_OR_FORCE_FRAME_ABI')
    valid=not reasons
    # No Fn/depth/HO selection: weak, negative and type3 remain in the sum.
    signed=np.full(3,np.nan,np.float64)
    if internal:signed=np.zeros(3,np.float64)
    elif external and valid:signed=(1. if inside[1] else -1.)*(frame.T@force[:3])
    category='HO' if external and (g1 in hand or g2 in hand) else 'other_external' if external else 'internal' if internal else 'not_object'
    return {'external':external,'internal':internal,'category':category,'constraint_force_evidence_valid':valid,'reasons':reasons,
        'signed_force_world_N':signed,'raw_force6':force,'raw_type':typ,'raw_worldid':world,'raw_slots':slots.tolist(),
        'host_efc_address_reordered':int(contact['efc_address']),'ordered_geom':pair.tolist(),'normal_force_selection_applied':False}


def impulse_terms(mass,vpre,vpost,dt,force,gravity,np):
    """FP32 stored velocities convert BEFORE subtraction; N*s then N."""
    a=np.asarray(vpre);b=np.asarray(vpost)
    require(a.dtype==b.dtype==np.float32 and a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all(),'Finite original free translation FP32')
    d=float(np.float32(dt));require(d>0 and math.isfinite(d),'Positive source WP32 dt')
    dp=mass*(b.astype(np.float64)-a.astype(np.float64));fg=mass*np.asarray(gravity,np.float64)
    jc=d*np.asarray(force,np.float64);jg=d*fg;residual=dp-jc-jg
    return dp,fg,jc,jg,residual,residual/d


def saved_velocity_clock(s,nv,vadr,clock,np):
    """Validate the full stored velocity, not just the object translation slice."""
    require(nv==26 and type(vadr) is int and 0<=vadr<=nv-6,'Source full26 freebody address')
    a=np.asarray(s['pre_qvel'],np.float32);b=np.asarray(s['post_qvel'],np.float32)
    require(a.shape==b.shape==(nv,) and np.isfinite(a).all() and np.isfinite(b).all(),'Finite original full qvel26')
    k,sub,pre,post=(s[x] for x in ('phase','substep','canonical_pre_tick','canonical_post_tick'))
    require(all(type(x) is int for x in (k,sub,pre,post)) and 0<=k<150 and 1<=sub<=4 and 0<=pre<post<=600 and post==pre+1 and pre==4*k+sub-1,'Adjacent bounded source ticks/phase/substep')
    times=[float(s[x]) for x in ('pre_time','post_time')]
    require(all(math.isfinite(x) for x in times) and all(float(np.float32(x))==x for x in times),'Original finite FP32 times')
    require(np.float32(times[0]).tobytes()==clock[pre].tobytes() and np.float32(times[1]).tobytes()==clock[post].tobytes(),'Source WP32 clock mismatch, no dt replacement')
    require(math.isfinite(float(s['external_force_max'])) and float(s['external_force_max'])>=0,'Finite saved external-force maximum')
    return a[vadr:vadr+3].copy(),b[vadr:vadr+3].copy()


def jsonable(v):
    if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    if hasattr(v,'tolist'):return jsonable(v.tolist())
    if type(v) is float and not math.isfinite(v):return None
    return v


def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    counts={'passive_MJB_attempted':0,'passive_MJB_returned':0,'saved_rows_read':0,'MjData_FK_forward_step_Torch_model_GPU':0};phase='metadata'
    try:
        import numpy as np
        import mujoco
        counts['passive_MJB_attempted']+=1;model=mujoco.MjModel.from_binary_path(str(checked(i['nominal_model'])));counts['passive_MJB_returned']+=1
        metadata,constants=inspect_constants(model,mujoco,np)
        with np.load(checked(i['actual_context']),allow_pickle=False) as f:context={k:f[k].copy() for k in f.files}
        readbacks={}
        for path in ('A0','A2'):
            with np.load(checked(i[path+'_actual_readbacks']),allow_pickle=False) as f:readbacks[path]={k:f[k].copy() for k in f.files}
        body=metadata['object_body_id'];mass=actual_mass(context,readbacks,body,np);metadata['actual_WP_mass_kg']=mass
        catalog=read(checked(i['geometry_catalog']));require(catalog['native_ngeom']==metadata['ngeom'] and catalog['native_object_body_id']==body,'Complete catalog same object')
        require(all(int(model.geom_bodyid[g['id']])==g['native_body_id'] and model.geom(g['id']).name==g['name'] for g in catalog['native_geom_catalog']),'MJB/catalog full native IDs')
        objects=set(metadata['object_geom_ids']);hand={g['id'] for g in catalog['native_hand_geoms']}
        constants.update(actual_body_mass_FP32=context['body_mass'],**{path+'__'+k:v for path,values in readbacks.items() for k,v in values.items()})
        np.savez_compressed(out/'passive_metadata_and_actual_readbacks.npz',**constants);c._write_json(out/'metadata.json',jsonable(metadata))
        phase='all_original_saved_rows'
        samples=[json.loads(s) for s in checked(i['physical_samples']).read_text(encoding='utf-8').splitlines()];counts['saved_rows_read']=len(samples);require(len(samples)==1808,'All1808 samples')
        identity_counts={'mode':{q:sum(s['mode']==q for s in samples) for q in ('donor','baseline','direction')},
            'path':{q:sum(s['path']==q for s in samples) for q in ('A0','A2')}}
        require(identity_counts['mode']=={'donor':1200,'baseline':96,'direction':512} and identity_counts['path']=={'A0':904,'A2':904},'Original whole population')
        identity_counts['path_mode_phase_substep']={}
        for s in samples:
            key='/'.join(map(str,(s['path'],s['mode'],s['phase'],s['substep'])))
            identity_counts['path_mode_phase_substep'][key]=identity_counts['path_mode_phase_substep'].get(key,0)+1
        clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+np.float32(metadata['dt_wp32_s'])))
        rows=[];long=[];data={k:[] for k in ('mass_kg_F64','dt_wp32_s_FP32','dt_used_s_F64','vpre_world_FP32','vpost_world_FP32','delta_p_Ns_F64',
            'FHO_N_F64','Fother_N_F64','Ftotal_N_F64','Fgravity_N_F64','F_known_valid_contacts_N_F64','Jcontact_Ns_F64','Jgravity_Ns_F64',
            'unassigned_known_terms_Ns_F64','closure_residual_Ns_F64','force_residual_N_F64','COM0velocity_valid','contact_evidence_complete','HO_complete','other_complete','gravity_known','noncontact_metadata_closed','closure_interpretation_eligible')}
        gravity=np.asarray(metadata['gravity_effective_world_m_s2'],np.float64);dt=np.float32(metadata['dt_wp32_s']);vadr=metadata['dofadr']
        for row_id,s in enumerate(samples):
            require(s['nacon']==len(s['contacts'])==len(s['raw_wp_contact']['geom']),'Contact count/order ABI')
            raw=s['raw_wp_contact'];require(all(len(raw[k])==s['nacon'] for k in ('type','worldid','efc_address')),'Raw contact columns')
            sums={'HO':np.zeros(3),'other_external':np.zeros(3)};complete={'HO':True,'other_external':True};valid_sum=np.zeros(3);external_count=internal_count=0
            for ci,contact in enumerate(s['contacts']):
                rawrow={k:raw[k][ci] for k in ('geom','type','worldid','efc_address')};q=contact_force(contact,rawrow,objects,hand,s['nefc'],np)
                if q['external']:
                    external_count+=1
                    if q['constraint_force_evidence_valid']:sums[q['category']]+=q['signed_force_world_N'];valid_sum+=q['signed_force_world_N']
                    else:complete[q['category']]=False
                if q['internal']:internal_count+=1
                long.append({'row_id':row_id,'original_contact_index':ci,'path':s['path'],'mode':s['mode'],'branch_id':s['branch_id'],**q})
            force_complete=all(complete.values());FHO=sums['HO'] if complete['HO'] else np.full(3,np.nan);Fother=sums['other_external'] if complete['other_external'] else np.full(3,np.nan)
            F=FHO+Fother if force_complete else np.full(3,np.nan)
            pre,post=saved_velocity_clock(s,metadata['nv'],vadr,clock,np)
            com0=metadata['COM0_isolated_freebody'];known_external=(s['external_force_max']==0.)
            dp,fg,jc,jg,unassigned,ef=impulse_terms(mass,pre,post,dt,F,gravity,np)
            eligible=bool(com0 and force_complete and metadata['noncontact_metadata_closed'] and known_external)
            if not com0:dp=np.full(3,np.nan);unassigned=np.full(3,np.nan)
            res=unassigned if eligible else np.full(3,np.nan);force_res=res/float(dt)
            record={'row_id':row_id,'path':s['path'],'mode':s['mode'],'branch_id':s['branch_id'],'phase':s['phase'],'substep':s['substep'],
                'canonical_pre_tick':s['canonical_pre_tick'],'canonical_post_tick':s['canonical_post_tick'],'pre_time':s['pre_time'],'post_time':s['post_time'],
                'mass_kg_F64':mass,'dt_wp32_s_FP32':dt,'dt_used_s_F64':float(dt),'vpre_world_FP32':pre,'vpost_world_FP32':post,
                'delta_p_Ns_F64':dp,'FHO_N_F64':FHO,'Fother_N_F64':Fother,'Ftotal_N_F64':F,'Fgravity_N_F64':fg,'F_known_valid_contacts_N_F64':valid_sum,
                'Jcontact_Ns_F64':jc,'Jgravity_Ns_F64':jg,'unassigned_known_terms_Ns_F64':unassigned,'closure_residual_Ns_F64':res,'force_residual_N_F64':force_res,
                'COM0velocity_valid':com0,'contact_evidence_complete':force_complete,'HO_complete':complete['HO'],'other_complete':complete['other_external'],
                'gravity_known':True,'noncontact_metadata_closed':metadata['noncontact_metadata_closed'],'closure_interpretation_eligible':eligible,
                'external_object_contacts':external_count,'internal_object_pairs_canceled':internal_count,'global_clearance':'UNKNOWN',
                'phase_scope':'contacts/object_solver_COM/pose solverPRE; free qvel delta PRE->POST within same5ms substep',
                'object_solver_COM':s['object_solver_COM'],'object_solver_pose':s['object_solver_pose'],'no_positive_Fn_selection':True}
            rows.append(record)
            for k in data:data[k].append(record[k])
        arrays={k:np.asarray(v,dtype=np.bool_ if k in ('COM0velocity_valid','contact_evidence_complete','HO_complete','other_complete','gravity_known','noncontact_metadata_closed','closure_interpretation_eligible') else np.float32 if k.endswith('FP32') else np.float64) for k,v in data.items()}
        np.savez_compressed(out/'all1808_momentum_arrays.npz',**arrays)
        np.savez_compressed(out/'all_contact_force_arrays.npz',row_id=np.asarray([x['row_id'] for x in long],np.int64),contact_index=np.asarray([x['original_contact_index'] for x in long],np.int64),
            signed_force_world_N=np.asarray([x['signed_force_world_N'] for x in long],np.float64).reshape(-1,3),external=np.asarray([x['external'] for x in long],bool),internal=np.asarray([x['internal'] for x in long],bool),evidence_valid=np.asarray([x['constraint_force_evidence_valid'] for x in long],bool))
        for name,values in (('all1808_rows',rows),('all_generated_contacts',long)):
            with (out/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for x in values:f.write(json.dumps(jsonable(x),allow_nan=False)+'\n')
            with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]) if values else ['row_id']);writer.writeheader()
                for x in values:writer.writerow({k:json.dumps(jsonable(v)) if isinstance(v,(list,dict)) or hasattr(v,'tolist') else jsonable(v) for k,v in x.items()})
        summary=[]
        for path in ('ALL','A0','A2'):
            for mode in ('ALL','donor','baseline','direction'):
                idx=[j for j,s in enumerate(samples) if (path=='ALL' or s['path']==path) and (mode=='ALL' or s['mode']==mode)]
                eligible=[j for j in idx if arrays['closure_interpretation_eligible'][j]];value=arrays['force_residual_N_F64'][eligible]
                summary.append({'path':path,'mode':mode,'all_rows':len(idx),'eligible_rows':len(eligible),'UNKNOWN_rows':len(idx)-len(eligible),
                    'eligible_only_force_residual_component_RMS_N':float(np.sqrt(np.mean(value*value))) if eligible else None,
                    'eligible_only_force_residual_vector_RMS_N':float(np.sqrt(np.mean(np.sum(value*value,axis=-1)))) if eligible else None,
                    'accuracy_pass_threshold':None,'eligibility_is_not_accuracy':True})
        for v in i.values():checked(v)
        for v in m['sources'].values():checked(v)
        require(same(pin(protocol_path),m['protocol_pin']),'Protocol changed')
        report={'schema':'full151105_saved_object_linear_momentum_CPU_analysis:1','status':'COMPLETED_CONDITIONAL_SAVED_MOMENTUM_NO_ACCURACY_GATE',
            'protocol':m['protocol_pin'],'input_pins':i,'counts':counts,'identity_counts':identity_counts,'metadata':metadata,'summary':summary,'all1808_preserved':True,
            'all_generated_contact_rows':len(long),'no_positive_Fn_filter':True,'source_applied_forces_independent_raw_vectors_saved':False,
            'known_force_omissions':'effective_qacc/solvergrad/complete passive/constraint generalized terms not saved; residual has no unique causal attribution',
            'independent_full_pyramid_decode_qualified':False,'physical_force_or_gradient_qualified':False,'human_force_qualified':False,'global_clearance_qualified':False,
            'whole_hidden_state_qualified':False,'no_angular_momentum_or_wrench_claim':True,'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',jsonable(report));c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:c._write_json(out/'failure.json',{'schema':'full151105_momentum_failure:1','phase':phase,'error':repr(error),'counts':counts,'no_automatic_retry':True,'no_force_unknown_zero_repair':True})
        except BaseException:pass
        raise
