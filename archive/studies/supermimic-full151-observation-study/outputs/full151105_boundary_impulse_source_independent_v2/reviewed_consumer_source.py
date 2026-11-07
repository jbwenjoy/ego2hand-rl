"""REVIEWED_SAVED_TARGET_DATASET_NOT_PHYSICS_QUALIFIED. No model/backend.

Metadata is JSON/hash only. --execute is Root's separate NumPy-only boundary.
No state conversion, force/wrench re-decode, Actor, model, MJB, or new physics.
"""
from __future__ import annotations
import csv,json,math,sys,time,zipfile
from pathlib import Path
from . import response_calibration105_v1 as c
ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_saved_boundary_impulse_target_dataset_CPU_protocol:1'
ROLE='old152_restored_boundary_impulse_target_dataset'
PREFIX='full151105_saved_boundary_impulse_targets_CPU_'
PREREG=ROOT/'protocols/full151105_saved_boundary_impulse_target_dataset_CPU_prereg_v1.json'
PREREG_SHA='dc805ccaf22f02a04ae5ec42319ff8cc31ff58c5a1a577198e1e8cfad7fd27a4'
CONTRACT=ROOT/'docs/full151_boundary_impulse_target_dataset_source_contract_v1.md'
CONTRACT_SHA='736250610d3d9fd5b3cc39e7c714fe3dae17f9f0498b4811795d8f2ebc487cd4'
INPUTS={'dataset_contract','H1_report','H1_inputs','H1_prediction_xi','H1_identities','H1_saved25','response_raw152','response_metadata152','response_saved22',
        'calibration_fit152','calibration_protocol','physical_samples','reward_snapshots','linear_report','linear_arrays','linear_rows','linear_metadata',
        'angular_report','angular_arrays','angular_rows','angular_contacts','angular_inertia','angular_saved9','angular_source_inventory','qualified_GPU_packet','replay_saved29','actual_context'}
OWN={'consumer':ROOT/'src/full151_gradient/boundary_impulse_dataset105_v1.py','CLI':ROOT/'scripts/build_full151_boundary_impulse_targets_saved_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_boundary_impulse_targets_saved_cpu_v1.py','borrowed_IO':ROOT/'src/full151_gradient/response_calibration105_v1.py'}
PHASES=(0,49,99,149)


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE,'New boundary dataset namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False,'Explicit CPU/noGPU')
    require(p.get('accuracy_gate') is None and p.get('output_prefix')==PREFIX,'No scientific residual gate')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and type(p.get('source_pins')) is dict and p['source_pins'],'Exact27 inputs/source map')
    if execute:require(p['CPU_run_authorized'],'Root explicit saved-label authorization required')
    return p


def metadata_preflight(protocol_path,output=None):
    require(not any(x in sys.modules for x in ('torch','mujoco','warp')),'Fresh metadata process, no backend')
    p=validate_protocol(read(protocol_path));i=p['inputs'];pre=read(PREREG)
    require(same(p['preregistration'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Frozen scientific parent')
    require(set(i)==set(pre['inputs']) and all(same(v,pre['inputs'][k]) for k,v in i.items()),'Exact predeclared input identities')
    require(same(i['dataset_contract'],{'path':str(CONTRACT),'sha256':CONTRACT_SHA}),'Unchanged source contract')
    for value in i.values():checked(value)
    h=read(checked(i['H1_report']));a=read(checked(i['angular_report']));l=read(checked(i['linear_report']));cp=read(checked(i['calibration_protocol']))
    require(h['schema']=='full151105_state_prediction452_CPU_analysis:1' and h['prediction_rows']==452,'Completed H1 all452 parent')
    require(a['schema']=='full151105_saved_object_angular_momentum_CPU_analysis:1' and a['all1808_and9473_preserved'] is True and l['all1808_preserved'] is True,'Complete conditional impulse parents')
    require(same(a['input_pins']['linear_report'],i['linear_report']) and same(a['input_pins']['linear_metadata'],i['linear_metadata']),'Same angular/linear lineage')
    for name in ('physical_samples','actual_context'):
        require(same(h['input_pins'][name],i[name]) and same(a['input_pins'][name],i[name]) and same(l['input_pins'][name],i[name]),'Same physical run/context')
    require(same(h['input_pins']['reward_snapshots'],i['reward_snapshots']) and same(h['input_pins']['qualified_GPU_prestate1808'],i['qualified_GPU_packet']),'Same native frame/prestate authority')
    require(same(cp['inputs']['response_raw152'],i['response_raw152']) and same(cp['inputs']['response_metadata152'],i['response_metadata152']) and same(cp['inputs']['response_saved_independent22'],i['response_saved22']),'Original calibration fitting cohort')
    for alias,parent in (('H1_saved25','H1_report'),('angular_saved9','angular_report')):
        r=read(checked(i[alias]));c.allchecks(r);require(r['engineering_qualified'] is True and same(r['producer_report'],i[parent]),'Actual positive saved parent')
    rr=read(checked(i['response_saved22']));c.allchecks(rr);require(rr['engineering_qualified'] is True,'Original saved-response22')
    replay=read(checked(i['replay_saved29']));c.allchecks(replay)
    require(replay['state_replay_qualified'] is True and replay['future_scoring_inputs_qualified'] is True and same(replay['qualified_prestate'],i['qualified_GPU_packet']),'Qualified native packet')
    routes={'H1_inputs':(i['H1_report'],'matched_input_arrays.npz'),'H1_prediction_xi':(i['H1_report'],'prediction_arrays.npz'),
            'H1_identities':(i['H1_report'],'group_identities452.json'),'linear_arrays':(i['linear_report'],'all1808_momentum_arrays.npz'),
            'linear_rows':(i['linear_report'],'all1808_rows.jsonl'),'angular_arrays':(i['angular_report'],'all1808_angular_momentum_arrays.npz'),
            'angular_rows':(i['angular_report'],'all1808_rows.jsonl'),'angular_contacts':(i['angular_report'],'all9473_generated_contacts.jsonl'),
            'angular_inertia':(i['angular_report'],'actual_inertia_observer.npz'),'angular_source_inventory':(i['angular_report'],'source_inventory.json'),
            'calibration_fit152':(i['calibration_protocol'],'matched_fit_data152.npz')}
    for alias,(parent,name) in routes.items():require(checked(i[alias]).samefile(checked(parent).parent/name),'Actual artifact route: '+alias)
    require(checked(i['response_raw152']).samefile(Path(rr['run_dir'])/'all152_branch_raw.npz') and checked(i['response_metadata152']).samefile(Path(rr['run_dir'])/'all152_branch_metadata.json'),'Actual old response artifacts')
    inventory=read(checked(i['angular_source_inventory']))
    for value in OWN.values():require(any(same(pin(value),v) for v in p['source_pins'].values()),'Current own/helper source bound')
    # Exactly one existing map; protocols are pinned references, never expanded.
    sources=c.canonical_sources(list(inventory.values())+list(p['source_pins'].values())+list(i.values())+[pin(PREREG)]+[h['protocol'],a['protocol'],l['protocol']])
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists(),'Fresh output')
    return {'status':'PASS_BOUNDARY_DATASET_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,'actual_arrays_read_label_math':0}


def declaration():
    return {'schema':'full151105_boundary_impulse_dataset_declaration:1','status':'REVIEWED_SAVED_TARGET_DATASET_NOT_PHYSICS_QUALIFIED','protocol_schema':SCHEMA,'role':ROLE,
        'input_aliases':sorted(INPUTS),'preregistration':pin(PREREG),'dataset_contract':pin(CONTRACT),'source_pins':{k:pin(v) for k,v in OWN.items()},
        'budget':{'groups':152,'source_substeps':608,'Torch_backend_MJB_model_physics':0},'actual_arrays_read_label_math':0}


def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def f32(a,shape,np):
    require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite original FP32 '+str(shape));return a
def exact_f32(value,shape,np):
    original=np.asarray(value);a=np.asarray(value,np.float32)
    require(a.shape==shape and np.isfinite(a).all() and np.array_equal(a.astype(np.float64),original.astype(np.float64)),'Original FP32 representability/shape');return a


def group_plan(identities,metadata,np):
    require(type(identities) is list and len(identities)==452 and type(metadata) is list and len(metadata)==152,'All452/152 identities')
    plan=[]
    for bid,row in enumerate(metadata):
        s,off=divmod(bid,19);pi,si=divmod(s,4);path=('A0','A2')[pi];k=PHASES[si];mode='baseline' if off<3 else 'direction';g=226*pi+150+19*si+off
        require(row['row_id']==bid and row['snapshot_id']==s and row['path']==path and row['k']==k and row['type']==mode and row['group_id']==g,'Original branch/group join')
        h=identities[g]
        require(h['group_id']==g and h['path']==path and h['mode']==mode and h['k']==k and h['branch_id']==bid and h['first_physical_pre_index']==4*g and h['last_physical_index']==4*g+3 and h['post20_GPU_row_index']==1808+g,'Original H1 source ordering')
        require(h['terminal_observation']==k+1 and h['valid_next_policy_decision']==(k+1<150),'Terminal150 response, no extra decision')
        if off>=3:
            ei,jr=divmod(off-3,8);j,signbit=divmod(jr,2)
            require(row['j']==j and row['epsilon_index']==ei and row['joint_index']==(0,4,8,12)[j] and row['sign']==(-1,1)[signbit],'EPS-major branch metadata')
        plan.append({'row_id':bid,'snapshot_id':s,'path':path,'mode':mode,'k':k,'branch_id':bid,'group_id':g,'physical_rows':[4*g+x for x in range(4)],'terminal_observation':k+1})
    return plan,np.asarray([r['group_id'] for r in plan],np.int64),np.asarray([r['physical_rows'] for r in plan],np.int64)


def training_join(h1,pred,raw,fit,gids,context,np):
    for key,shape in (('pre_x452_FP32',(452,58)),('target_x452_FP32',(452,58)),('actual_u452_FP32',(452,20)),('all_substep_u452x4x20_FP32',(452,4,20))):f32(h1[key],shape,np)
    x=h1['pre_x452_FP32'][gids].copy();u=h1['actual_u452_FP32'][gids].copy();y=h1['target_x452_FP32'][gids].copy()
    for key in ('pre_x452_FP32','target_x452_FP32','actual_u452_FP32'):require(bits(h1[key],pred[key]),'H1 matched/prediction input bytes')
    for value,key,fitkey,shape in ((x,'pre_visible152_FP32','pre',(8,19,58)),(u,'actual_u152_FP32','u',(8,19,20)),(y,'target_y152_FP32','target',(8,19,58))):
        f32(raw[key],shape,np);f32(fit[fitkey],value.shape,np)
        require(bits(value,raw[key].reshape(value.shape)) and bits(value,fit[fitkey]),'Actual per-group fit/native source bytes, not snapshot repetition')
    require(bits(h1['all_substep_u452x4x20_FP32'][gids],np.broadcast_to(u[:,None],(152,4,20)).copy()),'All4 actual absolute controls held')
    xi=f32(pred['physical_xi46_FP32'],(46,),np);cx=context['reader_context'];require(cx.dtype==np.float32 and cx.ndim==2 and cx.shape[0]>0 and cx.shape[1]==46 and np.isfinite(cx).all(),'Fixed physical xi source')
    require(bits(cx,np.broadcast_to(xi,cx.shape).copy()),'Native physical context all46 bytes')
    return {'x58_FP32':x,'u20_FP32':u,'target58_FP32':y,'physical_xi46_FP32':xi.copy()}


def rotate_and_residual(x,y,P,Jworld,Kworld,mass,Ibody,gravity_world,dt,np):
    """Saved native rawR9 observer; no projection/inverse and no zero defect."""
    n=len(x);B=np.swapaxes(P.astype(np.float64),-1,-2)
    Jp=np.einsum('nij,ncj->nci',B,Jworld);Kp=np.einsum('nij,ncj->nci',B,Kworld);gp=np.einsum('nij,j->ni',B,gravity_world)
    R0=x[:,43:52].astype(np.float64).reshape(n,3,3);R1=y[:,43:52].astype(np.float64).reshape(n,3,3)
    I0=R0@Ibody@np.swapaxes(R0,-1,-2);I1=R1@Ibody@np.swapaxes(R1,-1,-2)
    rv=y[:,52:55].astype(np.float64)-x[:,52:55].astype(np.float64)-(Jp.sum(axis=1)+mass*gp*dt[:,None])/mass
    momentum=np.einsum('nij,nj->ni',I0,x[:,55:58].astype(np.float64))+Kp.sum(axis=1)
    solved=np.full((n,3),np.nan,np.float64);nonsingular=np.zeros(n,bool)
    for row in range(n):
        try:solved[row]=np.linalg.solve(I1[row],momentum[row]);nonsingular[row]=True
        except np.linalg.LinAlgError:pass
    rw=y[:,55:58].astype(np.float64)-solved
    solve_residual=np.einsum('nij,nj->ni',I1,solved)-momentum
    finite=np.isfinite(rv).all(axis=1)&np.isfinite(rw).all(axis=1)&np.isfinite(solve_residual).all(axis=1)
    return {'J_palm_F64':Jp,'K_palm_F64':Kp,'palm_transpose_F64':B,'gravity_palm_F64':gp,'native_R0_raw_F64':R0,'native_R1_raw_F64':R1,
        'I_palm0_F64':I0,'I_palm1_F64':I1,'native_R0_det_F64':np.linalg.det(R0),'native_R1_det_F64':np.linalg.det(R1),
        'native_R0_orthogonality_defect_F64':np.swapaxes(R0,-1,-2)@R0-np.eye(3),'native_R1_orthogonality_defect_F64':np.swapaxes(R1,-1,-2)@R1-np.eye(3),
        'I_palm1_condition_number_F64':np.linalg.cond(I1),'angular_momentum_RHS_palm_F64':momentum,'omega1_from_saved_impulses_F64':solved,
        'rawR_solve_residual_F64':solve_residual,'rawR_solve_nonsingular_mask':nonsingular,'rawR_solve_finite_mask':finite,
        'r_v_saved_m_s_F64':rv,'r_w_saved_rad_s_F64':rw}


def four_substep_impulses(h,F,tau,np):
    n=len(h);require(h.dtype==np.float32 and h.shape==(n,4) and np.isfinite(h).all() and (h>0).all(),'Native FP32 four-substep duration')
    require(F.shape==tau.shape==(n,4,2,3) and F.dtype==tau.dtype==np.float64 and np.isfinite(F).all() and np.isfinite(tau).all(),'Two-source saved F64 force/COMtorque operands')
    hd=h.astype(np.float64)
    return (hd[:,:,None,None]*F).sum(axis=1),(hd[:,:,None,None]*tau).sum(axis=1),hd.sum(axis=1)


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
        for name,v in m['sources'].items():z.write(checked(v),name)
    phase='saved_operands';counts={'enrolled_groups':0,'selected_substeps':0,'new_model_Torch_MJB_backend_physics':0};guards={};saved={}
    try:
        import numpy as np
        def npz(alias):
            with np.load(checked(i[alias]),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
        def jsonl(alias):return [json.loads(s) for s in checked(i[alias]).read_text(encoding='utf-8').splitlines() if s.strip()]
        identities=read(checked(i['H1_identities']));metadata=read(checked(i['response_metadata152']));plan,gids,indices=group_plan(identities,metadata,np)
        c._write_json(out/'group152_identities.json',plan)
        h1,pred,raw,fit,context=(npz(k) for k in ('H1_inputs','H1_prediction_xi','response_raw152','calibration_fit152','actual_context'))
        # Save original chosen cohort before any enrollment guard can fail.
        saved={'source_group_id152':gids,'source_substep_rows152x4':indices,'H1_pre_selected_FP32':h1['pre_x452_FP32'][gids],
            'H1_u_selected_FP32':h1['actual_u452_FP32'][gids],'H1_target_selected_FP32':h1['target_x452_FP32'][gids]}
        np.savez_compressed(out/'selected_raw_input_operands.npz',**saved)
        data=training_join(h1,pred,raw,fit,gids,context,np);guards['all152_actual_native_calibration_input_bytes']=True
        packet=npz('qualified_GPU_packet');f32(packet['native_GPU_prestate58'],(1808,58),np);f32(packet['palm_rot'],(1808,3,3),np);f32(packet['palm_pos'],(1808,3),np)
        require(packet['completed_mask'].shape==(1808,) and packet['completed_mask'].dtype==np.bool_ and packet['completed_mask'].all() and np.array_equal(packet['source_pre_row'],np.arange(1808)),'Complete original qualified packet')
        require(bits(packet['native_GPU_prestate58'][indices[:,0]],data['x58_FP32']),'Actual firstPRE native58 authority')
        samples=jsonl('physical_samples');lr=jsonl('linear_rows');ar=jsonl('angular_rows');contacts=jsonl('angular_contacts');rewards=jsonl('reward_snapshots')
        require(len(samples)==len(lr)==len(ar)==1808 and len(contacts)==9473 and len(rewards)==452,'Complete source populations')
        rewardmap={}
        for r in rewards:
            key=(r['path'],r['k'],r['branch_id']);require(key not in rewardmap,'Unique reward/source key');rewardmap[key]=r
        P=[];pp=[];preq=[];postq=[];prev=[];postv=[];held=[];row_identities=[]
        for bid,group in enumerate(plan):
            ids=indices[bid];rw=rewardmap[(group['path'],group['k'],bid)]
            pr=exact_f32(rw['state20_reader_palm_rot'],(1,3,3),np)[0];pos=exact_f32(rw['state20_reader_palm_pos'],(1,3),np)[0]
            require(bits(packet['palm_rot'][ids],np.broadcast_to(pr,(4,3,3)).copy()) and bits(packet['palm_pos'][ids],np.broadcast_to(pos,(4,3)).copy()),'Recorded fixed native Reader group frame, not independent1808mocap')
            ctrl=[];q0=[];q1=[];v0=[];v1=[]
            for sub,t in enumerate(ids,1):
                s=samples[int(t)];identity=(group['path'],group['mode'],bid,group['k'],sub)
                require((s['path'],s['mode'],s['branch_id'],s['phase'],s['substep'])==identity,'Original physical608 identity')
                for row in (lr[int(t)],ar[int(t)]):
                    require(row['row_id']==int(t) and (row['path'],row['mode'],row['branch_id'],row['phase'],row['substep'])==identity,'Impulse source row identity')
                    require(all(row[key]==s[key] for key in ('canonical_pre_tick','canonical_post_tick','pre_time','post_time')),'Impulse source ticks/times exact physical join')
                require(s['canonical_pre_tick']==4*group['k']+sub-1 and s['canonical_post_tick']==4*group['k']+sub,'Original source ticks')
                ctrl.append(exact_f32(s['ctrl'],(20,),np));q0.append(exact_f32(s['pre_qpos'],(27,),np));q1.append(exact_f32(s['post_qpos'],(27,),np));v0.append(exact_f32(s['pre_qvel'],(26,),np));v1.append(exact_f32(s['post_qvel'],(26,),np))
                row_identities.append({'group_row_id':bid,'source_row_id':int(t),'path':s['path'],'mode':s['mode'],'branch_id':bid,'k':s['phase'],'substep':sub})
            ctrl=np.asarray(ctrl);q0,q1,v0,v1=map(np.asarray,(q0,q1,v0,v1))
            require(bits(ctrl,np.broadcast_to(data['u20_FP32'][bid],(4,20)).copy()) and bits(ctrl,h1['all_substep_u452x4x20_FP32'][gids[bid]]),'Four held actualctrl bit join')
            require(bits(q0[1:],q1[:-1]) and bits(v0[1:],v1[:-1]),'Three internal PRE/POST full27/26 bit links')
            P.append(pr);pp.append(pos);preq.append(q0);postq.append(q1);prev.append(v0);postv.append(v1);held.append(ctrl)
        guards.update(all608_identity_and_tick_join=True,all152_controls_four_held=True,all456_internal_primitive_links=True,all152_recorded_fixed_group_frames=True)
        c._write_json(out/'selected608_row_identities.json',row_identities)
        la=npz('linear_arrays');aa=npz('angular_arrays');ib=npz('angular_inertia');lm=read(checked(i['linear_metadata']));body=lm['object_body_id']
        for name in ('body_inertia','body_iquat'):
            require(bits(ib['source_context__'+name],context[name]),'Angular full actual context source')
            require(bits(ib['actual_inertia_principal_FP32' if name=='body_inertia' else 'actual_iquat_raw_FP32'],context[name][0,body]),'Actual object inertia source')
        mass=float(lm['actual_WP_mass_kg']);require(math.isfinite(mass) and mass>0 and mass==float(context['body_mass'][0,body]),'Actual source FP32 mass')
        I=np.asarray(ib['I_body_F64']);require(I.dtype==np.float64 and I.shape==(3,3) and np.isfinite(I).all(),'Frozen angular inertia observer');np.linalg.cholesky(I)
        for key in ('contact_evidence_complete','HO_complete','other_complete','closure_interpretation_eligible'):
            for a in (la,aa):require(a[key].dtype==np.bool_ and a[key].shape==(1808,) and a[key][indices].all(),'All608 complete conditional source evidence; no sample filtering')
        require(bits(la['dt_wp32_s_FP32'],aa['dt_wp32_s_FP32']),'Same native source dt arrays')
        h=la['dt_wp32_s_FP32'][indices];require(h.dtype==np.float32 and h.shape==(152,4) and np.isfinite(h).all() and (h>0).all() and np.all(h==np.float32(lm['dt_wp32_s'])),'Native fixed FP32 substep dt')
        F=np.stack((la['FHO_N_F64'][indices],la['Fother_N_F64'][indices]),axis=2);tau=np.stack((aa['tau_HO_total_Nm_F64'][indices],aa['tau_other_total_Nm_F64'][indices]),axis=2)
        require(F.shape==tau.shape==(152,4,2,3) and F.dtype==tau.dtype==np.float64 and np.isfinite(F).all() and np.isfinite(tau).all(),'Complete frozen分源vectors')
        witness=np.zeros((152,2),bool);observed_count=np.zeros((152,4,2),np.int64);selected={int(t):(bid,sub) for bid,ids in enumerate(indices) for sub,t in enumerate(ids)};seen={int(t):set() for t in indices.flat}
        for q in contacts:
            t=q['row_id']
            if t not in selected:continue
            bid,sub=selected[t];ci=q['original_contact_index'];require(type(ci) is int and ci not in seen[t] and 0<=ci<samples[t]['nacon'],'Original contact index');seen[t].add(ci)
            require(q['path']==plan[bid]['path'] and q['mode']==plan[bid]['mode'] and q['branch_id']==bid,'Selected witness group identity')
            if q['external']:
                require(q['category'] in ('HO','other_external') and q['constraint_force_evidence_valid'] is True,'No UNKNOWN external witness enrollment')
                require(q['raw_type']&1 and q['raw_worldid']==0 and q['host_efc_address_reordered']>=0,'Source constraint witness, never Fn-positive mode')
                slots=np.asarray(q['raw_slots']);require(slots.ndim==1 and slots.size>0 and slots.dtype.kind in 'iu' and (slots>=0).any() and ((slots==-1)|((slots>=0)&(slots<samples[t]['nefc']))).all(),'Valid source raw slots')
                cat=0 if q['category']=='HO' else 1;witness[bid,cat]=True;observed_count[bid,sub,cat]+=1
        require(all(len(seen[t])==samples[t]['nacon'] for t in seen),'Every generated contact in all608 retained, no Fn selection')
        guards.update(all608_conditional_impulse_evidence=True,all608_generated_contacts_accounted=True,all_actual_mass_inertia_context_sources=True)
        J,K,dt=four_substep_impulses(h,F,tau,np);P=np.asarray(P,np.float32)
        gravity=np.asarray(lm['gravity_effective_world_m_s2'],np.float64);require(gravity.shape==(3,) and np.isfinite(gravity).all(),'Source effective WP32 gravity')
        operands={**data,'source_group_id152':gids,'source_substep_rows152x4':indices,'source_dt_FP32':h,'DeltaT_F64':dt,'F_world_substeps_F64':F,'tau_COM_world_substeps_F64':tau,
            'J_world_F64':J,'K_world_F64':K,'palm_rot_raw_FP32':P,'palm_pos_raw_FP32':np.asarray(pp,np.float32),'I_body_F64':I,'mass_actual_F64':np.asarray(mass),'gravity_world_source_F64':gravity,
            'raw_qpos_PRE152x4_FP32':np.asarray(preq),'raw_qpos_POST152x4_FP32':np.asarray(postq),'raw_qvel_PRE152x4_FP32':np.asarray(prev),'raw_qvel_POST152x4_FP32':np.asarray(postv),'actual_ctrl152x4_FP32':np.asarray(held),
            'boundary_constraint_witness':witness,'boundary_observed_external_contact_count152x4x2':observed_count,
            'angular_observer_group_defect_world_Nms_F64':aa['delta_L_minus_contact_J_Nms_F64'][indices].sum(axis=1),
            'linear_observer_group_defect_world_Ns_F64':la['unassigned_known_terms_Ns_F64'][indices].sum(axis=1)}
        np.savez_compressed(out/'all152_target_operands_before_residual.npz',**operands)
        phase='native_rawR_saved_defect';result=rotate_and_residual(data['x58_FP32'],data['target58_FP32'],P,J,K,mass,I,gravity,dt,np)
        targets={**operands,**result};np.savez_compressed(out/'all152_boundary_impulse_targets.npz',**targets)
        guards['all152_native_rawR_solves_finite']=bool(result['rawR_solve_nonsingular_mask'].all() and result['rawR_solve_finite_mask'].all())
        c._write_json(out/'enrollment_guards.json',guards)
        require(guards['all152_native_rawR_solves_finite'],'All152 finite/nonsingular rawR solves required; all raw operands/masks kept, no filtering')
        rows=[]
        for bid,group in enumerate(plan):rows.append({**group,'boundary_HO_constraint_witness':bool(witness[bid,0]),'boundary_nonhand_constraint_witness':bool(witness[bid,1]),
            'J_HO_palm_Ns':result['J_palm_F64'][bid,0].tolist(),'J_nonhand_palm_Ns':result['J_palm_F64'][bid,1].tolist(),'K_HO_palm_Nms':result['K_palm_F64'][bid,0].tolist(),'K_nonhand_palm_Nms':result['K_palm_F64'][bid,1].tolist(),
            'r_v_saved_m_s':result['r_v_saved_m_s_F64'][bid].tolist(),'r_w_saved_rad_s':result['r_w_saved_rad_s_F64'][bid].tolist()})
        c._write_json(out/'all152_target_rows.json',rows)
        with (out/'all152_target_rows.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        for v in i.values():checked(v)
        for v in m['sources'].values():checked(v)
        require(same(pin(protocol_path),m['protocol_pin']) and not any(x in sys.modules for x in ('torch','mujoco','warp')),'Unchanged input/protocol; no backend')
        counts.update(enrolled_groups=152,selected_substeps=608)
        report={'schema':'full151105_saved_boundary_impulse_target_dataset_CPU_analysis:1','status':'COMPLETED_ALL152_CONDITIONAL_TARGET_DATASET_NOT_PHYSICS_CERTIFICATE','protocol':m['protocol_pin'],'input_pins':i,'counts':counts,
            'enrollment_qualified':True,'guards':guards,'cohort':{'baseline_groups':24,'direction_groups':128,'donor_groups':0,'fresh_groups':0},
            'frame_scope':'452 recordedReader frames plus fixed-palm source/topology, NOT1808 direct mocap measurements; P.T rawFP32 promotedF64, no inverse/project',
            'saved_defect_scope':'Native raw58 R9/velocities; actual angular I_body F64 normalized-iquat observer and decoded source contact impulses. r_v/r_w never forced to0; not continuous exact laws.',
            'mode_scope':'boundary source constraint witness, not positiveFn/self15/human or global clearance','accuracy_gate':None,'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        c._write_json(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'guards_completed':guards,'enrollment_qualified':False,'no_repair_or_row_filter':True})
        c._write_json(out/'report.json',{'schema':'full151105_saved_boundary_impulse_target_dataset_CPU_analysis:1','status':'BLOCKED_DATASET_ENROLLMENT_ALL_COHORT_NO_REPAIR','enrollment_qualified':False,'protocol':m['protocol_pin'],'counts':counts,'guards_completed':guards,'error':repr(error)})
        raise
