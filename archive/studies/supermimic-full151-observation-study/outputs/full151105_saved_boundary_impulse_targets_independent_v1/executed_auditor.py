"""All152/608 independent saved targets; no producer math or backend calls.

Alternate scalar fsum/raw-basis multiplication and partial-pivot elimination.
F64 tolerance compares implementations; it never tests a defect against zero.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/boundary_impulse_independent_math_v1.py'
MATH_SHA='48eb17c899491f303d681c862fbaebb512f51f6eab5d10bc03118e01ecf8bef4'
UTIL=ROOT/'scripts/audit_object_angular_momentum_saved_cpu_v1.py'
UTIL_SHA='4240efb709dd97eefcecd2222af59bfbd9e73eea6cc55173935ebcc338b139bf'
PREREG=ROOT/'protocols/full151105_saved_boundary_impulse_target_dataset_CPU_prereg_v1.json'
PREREG_SHA='dc805ccaf22f02a04ae5ec42319ff8cc31ff58c5a1a577198e1e8cfad7fd27a4'
CORE=ROOT/'src/full151_gradient/boundary_impulse_dataset105_v1.py'
CORE_SHA='c7d7a82b8e56409b771efe6c2a8d08f8e2df37b591fb407d89275bf4a39b7553'
READY=ROOT/'outputs/full151105_boundary_impulse_dataset_preparation_CPU_v1/READY.json'
READY_SHA='8cc6b59c228fa3ba0ae806c60dbb51bfb283385de4e9d9fc4655a6f0d6713084'


def module(p,expected):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen independent code changed '+str(p))
    s=importlib.util.spec_from_file_location('boundary_saved_'+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


U=module(UTIL,UTIL_SHA)
require=U.require;pin=U.pin;sha=U.sha;read=U.read;checked=U.checked;same=U.same;load=U.load;bits=U.bits;dump=U.dump;eq=U.equivalent


def metadata(protocol_path):
    require(sha(PREREG)==PREREG_SHA and sha(CORE)==CORE_SHA and sha(READY)==READY_SHA,'Final method/source/proof')
    p,pr=read(protocol_path),read(PREREG)
    require(p['schema']=='full151105_saved_boundary_impulse_target_dataset_CPU_protocol:1' and p['role']==pr['role'] and p['accuracy_gate'] is None,'Descriptive dataset namespace')
    require(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False and p['output_prefix']==pr['output_prefix'],'Explicit CPU namespace')
    require(same(p['preregistration'],pin(PREREG)) and set(p['inputs'])==set(pr['inputs']) and len(p['inputs'])==27,'Exact27 aliases')
    for k,v in p['inputs'].items():require(same(v,pr['inputs'][k]),'Same fixed source '+k)
    for v in [*p['source_pins'].values(),*pr['source_pins'].values()]:checked(v)
    require(any(same(v,pin(CORE)) for v in p['source_pins'].values()),'Executed core bound')
    for alias in ('H1_saved25','response_saved22','angular_saved9','replay_saved29'):
        r=read(checked(p['inputs'][alias]));require(type(r['passed']) is int and r['passed']==r['total']==len(r['checks']) and all(v is True for v in r['checks'].values()),'Actual original all-true parent '+alias)
    ready=read(READY);proof=read(checked(ready['cpu_report']));require(proof['passed']==proof['total']==3 and all(v is True for v in proof['checks'].values()) and ready['source_pins']==proof['source_pins'],'Focused current-source producer3')
    for v in ready['source_pins'].values():checked(v)
    return p


def plan_record(r):
    return {'row_id':r['branch_id'],'snapshot_id':r['snapshot_id'],'path':r['path'],'mode':r['mode'],'k':r['k'],'branch_id':r['branch_id'],
      'group_id':r['group_id'],'physical_rows':r['substep_source_rows'],'terminal_observation':r['terminal_observation']}


def jsonl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def csv_matches(path,rows):
    with Path(path).open(newline='',encoding='utf8') as f:table=list(csv.DictReader(f))
    if len(table)!=len(rows):return False
    for got,wanted in zip(table,rows):
        for k,v in wanted.items():
            if k not in got:return False
            s=got[k]
            if isinstance(v,(list,dict)):ok=json.loads(s)==v
            elif v is None:ok=s==''
            elif type(v) is bool:ok=s==str(v)
            elif type(v) is int:ok=int(s)==v
            elif type(v) is float:ok=float(s)==v
            else:ok=s==v
            if not ok:return False
    return True


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_target_read.json',{'auditor':auditor,'math':pin(MATH),'F64_implementation_comparison':{'rtol':U.F64_RTOL,'atol':U.F64_ATOL,'not_a_defect_zero_or_accuracy_gate':True},'actual_target_values_read':False})
    p=metadata(protocol_path);i=p['inputs'];m=module(MATH,MATH_SHA);r=read(run/'report.json');checks={};detail={};diff={}
    def gate(k,v):checks[k]=bool(v)
    def compare(k,got,wanted):
        a,b=np.asarray(got),np.asarray(wanted);ok=eq(a,b)
        if a.shape==b.shape:
            valid=np.isfinite(a)&np.isfinite(b)
            if valid.any():diff[k]=max(diff.get(k,0.),float(abs(a[valid]-b[valid]).max()))
        return ok
    sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json')
    for v in [*sources.values(),*outputs.values()]:checked(v)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('authorized_source_samefile_SHA_ZIP_outputpins_and_all27aliases',p['CPU_run_authorized'] is True and same(r['protocol'],pin(protocol_path)) and p==read(run/'protocol.json') and r['input_pins']==i and zipok)
    h1=load(checked(i['H1_inputs']));pred=load(checked(i['H1_prediction_xi']));raw=load(checked(i['response_raw152']));fit=load(checked(i['calibration_fit152']));ctx=load(checked(i['actual_context']));packet=load(checked(i['qualified_GPU_packet']))
    schedule=m.expected_restored_schedule();plan=[plan_record(x) for x in schedule];gids=np.array([x['group_id'] for x in schedule],dtype='i8');indices=np.array([x['substep_source_rows'] for x in schedule],dtype='i8')
    identities=read(checked(i['H1_identities']));oldmeta=read(checked(i['response_metadata152']));require(len(identities)==452 and len(oldmeta)==152,'Original population')
    d=load(run/'all152_boundary_impulse_targets.npz');operands=load(run/'all152_target_operands_before_residual.npz');selected=load(run/'selected_raw_input_operands.npz')
    joins=True
    for x in schedule:
        bid,g=x['branch_id'],x['group_id'];h,old=identities[g],oldmeta[bid]
        joins &= h['group_id']==g and (h['path'],h['mode'],h['k'],h['branch_id'])==(x['path'],x['mode'],x['k'],bid) and h['first_physical_pre_index']==4*g and h['last_physical_index']==4*g+3 and h['post20_GPU_row_index']==1808+g
        joins &= old['row_id']==bid and old['snapshot_id']==x['snapshot_id'] and old['group_id']==g and old['type']==x['mode'] and old['path']==x['path'] and old['k']==x['k']
        if x['mode']=='direction':joins &= all(old[k]==x[k] for k in ('j','joint_index','epsilon_index','sign'))
    require(joins,'Exact old152 EPS-major identities, never fresh snapshot repeat')
    expected={'x58_FP32':h1['pre_x452_FP32'][gids],'u20_FP32':h1['actual_u452_FP32'][gids],'target58_FP32':h1['target_x452_FP32'][gids],'physical_xi46_FP32':pred['physical_xi46_FP32'],'source_group_id152':gids,'source_substep_rows152x4':indices}
    for k,shape in [('x58_FP32',(152,58)),('target58_FP32',(152,58)),('u20_FP32',(152,20)),('physical_xi46_FP32',(46,))]:require(expected[k].dtype==np.float32 and expected[k].shape==shape and np.isfinite(expected[k]).all(),'Original FP32 data '+k)
    for key,hkey in [('x58_FP32','pre_x452_FP32'),('u20_FP32','actual_u452_FP32'),('target58_FP32','target_x452_FP32')]:
        require(bits(h1[hkey],pred[hkey]),'Original H1 input copies')
        rawkey,fitkey={'x58_FP32':('pre_visible152_FP32','pre'),'u20_FP32':('actual_u152_FP32','u'),'target58_FP32':('target_y152_FP32','target')}[key]
        require(bits(expected[key],raw[rawkey].reshape(expected[key].shape)) and bits(expected[key],fit[fitkey]),'Actual native/old fit152 source copies')
    require(bits(ctx['reader_context'],np.broadcast_to(expected['physical_xi46_FP32'],ctx['reader_context'].shape).copy()),'Fixed xi46 full bytes')
    require(packet['completed_mask'].dtype==bool and packet['completed_mask'].shape==(1808,) and packet['completed_mask'].all() and np.array_equal(packet['source_pre_row'],np.arange(1808)),'Qualified native packet row authority')
    require(bits(packet['native_GPU_prestate58'][indices[:,0]],expected['x58_FP32']),'Each group actual native firstPRE')
    expected_selected={'source_group_id152':gids,'source_substep_rows152x4':indices,'H1_pre_selected_FP32':expected['x58_FP32'],'H1_u_selected_FP32':expected['u20_FP32'],'H1_target_selected_FP32':expected['target58_FP32']}
    gate('all152_original_groups_native58_fit_copies_xi46_and_saved_raw_indices',plan==read(run/'group152_identities.json') and all(bits(v,d[k]) for k,v in expected.items()) and set(selected)==set(expected_selected) and all(bits(v,selected[k]) for k,v in expected_selected.items()) and all(bits(v,d[k]) for k,v in operands.items()))
    samples=jsonl(checked(i['physical_samples']));lr=jsonl(checked(i['linear_rows']));ar=jsonl(checked(i['angular_rows']));rewards=jsonl(checked(i['reward_snapshots']));contacts=jsonl(checked(i['angular_contacts']))
    require(len(samples)==len(lr)==len(ar)==1808 and len(rewards)==452 and len(contacts)==9473,'Complete fixed source populations')
    rewardmap={(x['path'],x['k'],x['branch_id']):x for x in rewards};require(len(rewardmap)==452,'Unique reward rows')
    subidentities=[];P=[];pos=[];preq=[];postq=[];prev=[];postv=[];control=[]
    for x in schedule:
        bid=x['branch_id'];ids=indices[bid];reward=rewardmap[(x['path'],x['k'],bid)]
        pr=np.asarray(reward['state20_reader_palm_rot'],dtype='f4')[0];pp=np.asarray(reward['state20_reader_palm_pos'],dtype='f4')[0]
        require(bits(packet['palm_rot'][ids],np.broadcast_to(pr,(4,3,3)).copy()) and bits(packet['palm_pos'][ids],np.broadcast_to(pp,(4,3)).copy()),'All4 fixed recorded frame bytes')
        q0=[];q1=[];v0=[];v1=[];u=[]
        for sub,t in enumerate(ids,1):
            s=samples[int(t)];identity=(x['path'],x['mode'],bid,x['k'],sub)
            require((s['path'],s['mode'],s['branch_id'],s['phase'],s['substep'])==identity and s['canonical_pre_tick']==4*x['k']+sub-1 and s['canonical_post_tick']==4*x['k']+sub,'Original selected physical identity/ticks')
            for table in (lr,ar):require(table[int(t)]['row_id']==int(t) and all(table[int(t)][k]==s[k] for k in ('path','mode','branch_id','phase','substep','canonical_pre_tick','canonical_post_tick','pre_time','post_time')),'Original impulse tables match row/time')
            q0.append(np.asarray(s['pre_qpos'],dtype='f4'));q1.append(np.asarray(s['post_qpos'],dtype='f4'));v0.append(np.asarray(s['pre_qvel'],dtype='f4'));v1.append(np.asarray(s['post_qvel'],dtype='f4'));u.append(np.asarray(s['ctrl'],dtype='f4'))
            subidentities.append({'group_row_id':bid,'source_row_id':int(t),'path':x['path'],'mode':x['mode'],'branch_id':bid,'k':x['k'],'substep':sub})
        q0,q1,v0,v1,u=map(np.asarray,(q0,q1,v0,v1,u));require(bits(q0[1:],q1[:-1]) and bits(v0[1:],v1[:-1]),'All456 within-group primitive links')
        require(bits(u,np.broadcast_to(expected['u20_FP32'][bid],(4,20)).copy()) and bits(u,h1['all_substep_u452x4x20_FP32'][gids[bid]]),'All608 held actual controls')
        P.append(pr);pos.append(pp);preq.append(q0);postq.append(q1);prev.append(v0);postv.append(v1);control.append(u)
    raw_saved={'palm_rot_raw_FP32':np.asarray(P),'palm_pos_raw_FP32':np.asarray(pos),'raw_qpos_PRE152x4_FP32':np.asarray(preq),'raw_qpos_POST152x4_FP32':np.asarray(postq),'raw_qvel_PRE152x4_FP32':np.asarray(prev),'raw_qvel_POST152x4_FP32':np.asarray(postv),'actual_ctrl152x4_FP32':np.asarray(control)}
    gate('all608_source_ticks_times_456_links_heldcontrols_and_rawP_frame_bytes',subidentities==read(run/'selected608_row_identities.json') and all(bits(v,d[k]) for k,v in raw_saved.items()))
    la=load(checked(i['linear_arrays']));aa=load(checked(i['angular_arrays']));ib=load(checked(i['angular_inertia']));lm=read(checked(i['linear_metadata']));body=lm['object_body_id']
    for key in ('contact_evidence_complete','HO_complete','other_complete','closure_interpretation_eligible'):
        for source in (la,aa):m.require_complete_dataset(source[key][indices])
    for name in ('body_inertia','body_iquat'):require(bits(ib['source_context__'+name],ctx[name]) and bits(ib['actual_inertia_principal_FP32' if name=='body_inertia' else 'actual_iquat_raw_FP32'],ctx[name][0,body]),'Actual fixed normalized-inertia observer lineage')
    mass=float(lm['actual_WP_mass_kg']);require(mass==float(ctx['body_mass'][0,body]),'Actual mass')
    h=la['dt_wp32_s_FP32'][indices];require(bits(la['dt_wp32_s_FP32'],aa['dt_wp32_s_FP32']),'Native timestep sources same bytes')
    F=np.stack((la['FHO_N_F64'][indices],la['Fother_N_F64'][indices]),2);T=np.stack((aa['tau_HO_total_Nm_F64'][indices],aa['tau_other_total_Nm_F64'][indices]),2);g=np.asarray(lm['gravity_effective_world_m_s2'],dtype='f8');I=ib['I_body_F64']
    source_operands={'source_dt_FP32':h,'F_world_substeps_F64':F,'tau_COM_world_substeps_F64':T,'I_body_F64':I,'mass_actual_F64':np.asarray(mass),'gravity_world_source_F64':g,
      'angular_observer_group_defect_world_Nms_F64':aa['delta_L_minus_contact_J_Nms_F64'][indices].sum(axis=1),'linear_observer_group_defect_world_Ns_F64':la['unassigned_known_terms_Ns_F64'][indices].sum(axis=1)}
    gate('all608_complete_conditional_source_vectors_actual_mass_inertia_and_dt',all(bits(v,d[k]) for k,v in source_operands.items()))
    witness=np.zeros((152,2),bool);counts=np.zeros((152,4,2),dtype='i8');selected_rows={int(t):(bid,sub) for bid,ids in enumerate(indices) for sub,t in enumerate(ids)};seen={t:set() for t in selected_rows}
    for q in contacts:
        t=q['row_id']
        if t not in selected_rows:continue
        bid,sub=selected_rows[t];ci=q['original_contact_index'];require(type(ci) is int and 0<=ci<samples[t]['nacon'] and ci not in seen[t],'Original contact index');seen[t].add(ci)
        require((q['path'],q['mode'],q['branch_id'])==(plan[bid]['path'],plan[bid]['mode'],bid),'Source contact owner')
        if q['external']:
            require(q['constraint_force_evidence_valid'] is True and q['category'] in ('HO','other_external'),'Unknown external blocks whole dataset')
            slots=np.asarray(q['raw_slots']);require(q['raw_type']&1 and q['raw_worldid']==0 and q['host_efc_address_reordered']>=0 and slots.dtype.kind in 'iu' and slots.ndim==1 and slots.size>0 and (slots>=0).any() and ((slots==-1)|((slots>=0)&(slots<samples[t]['nefc']))).all(),'Original valid constraint witness, no Fn test')
            require(all(np.isfinite(np.asarray(q[k],dtype='f8')).all() for k in ('raw_force6','frame_saved','position_world_saved','COM_solver_PRE_saved')),'Finite original witness operands')
            c=0 if q['category']=='HO' else 1;witness[bid,c]=True;counts[bid,sub,c]+=1
    require(all(len(seen[t])==samples[t]['nacon'] for t in seen),'All generated contacts in608 retain original indices')
    gate('constraint_source_witness_and_all_contact_indices_noFn_or_self15_filter',bits(witness,d['boundary_constraint_witness']) and bits(counts,d['boundary_observed_external_contact_count152x4x2']))
    values={};numeric=True
    rename={'J_world_Ns_F64':'J_world_F64','K_world_Nms_F64':'K_world_F64','J_palm_Ns_F64':'J_palm_F64','K_palm_Nms_F64':'K_palm_F64','gravity_palm_m_s2_F64':'gravity_palm_F64','DeltaT_s_F64':'DeltaT_F64','I_p0_F64':'I_palm0_F64','I_p1_F64':'I_palm1_F64','angular_RHS_F64':'angular_momentum_RHS_palm_F64','angular_solved_omega_F64':'omega1_from_saved_impulses_F64','angular_solve_residual_F64':'rawR_solve_residual_F64','r_v_saved_m_s_F64':'r_v_saved_m_s_F64','r_w_saved_rad_s_F64':'r_w_saved_rad_s_F64','raw_R0_det_F64':'native_R0_det_F64','raw_R1_det_F64':'native_R1_det_F64','raw_R0_orthogonality_residual_F64':'native_R0_orthogonality_defect_F64','raw_R1_orthogonality_residual_F64':'native_R1_orthogonality_defect_F64'}
    for bid in range(152):
        calculated=m.four_substep_labels(expected['x58_FP32'][bid],expected['target58_FP32'][bid],raw_saved['palm_rot_raw_FP32'][bid],h[bid],F[bid],T[bid],mass,g,I)
        for own,key in rename.items():
            value=calculated[own];numeric &= compare(key,d[key][bid],value);values.setdefault(key,[]).append(value)
    numeric &= bits(d['palm_transpose_F64'],raw_saved['palm_rot_raw_FP32'].astype('f8').swapaxes(-1,-2)) and bits(d['native_R0_raw_F64'],expected['x58_FP32'][:,43:52].astype('f8').reshape(152,3,3)) and bits(d['native_R1_raw_F64'],expected['target58_FP32'][:,43:52].astype('f8').reshape(152,3,3))
    m.require_complete_dataset(d['rawR_solve_nonsingular_mask'],d['rawR_solve_finite_mask'])
    gate('all152_fsum_fourstep_rawPtranspose_nativeR9_alternate_solve_defects',numeric)
    targetrows=read(run/'all152_target_rows.json');tables=True
    for bid,row in enumerate(targetrows):
        tables &= all(row[k]==v for k,v in plan[bid].items()) and row['boundary_HO_constraint_witness']==bool(witness[bid,0]) and row['boundary_nonhand_constraint_witness']==bool(witness[bid,1])
        for key,array in [('J_HO_palm_Ns','J_palm_F64'),('J_nonhand_palm_Ns','J_palm_F64'),('K_HO_palm_Nms','K_palm_F64'),('K_nonhand_palm_Nms','K_palm_F64')]:tables &= compare(key,row[key],d[array][bid,0 if 'HO' in key else 1])
        for key,array in [('r_v_saved_m_s','r_v_saved_m_s_F64'),('r_w_saved_rad_s','r_w_saved_rad_s_F64')]:tables &= compare(key,row[key],d[array][bid])
    guards=read(run/'enrollment_guards.json')
    gate('all152_target_rows_and_whole_enrollment_not_accuracy_or_physics',len(targetrows)==152 and tables and csv_matches(run/'all152_target_rows.csv',targetrows) and guards==r['guards'] and all(v is True for v in guards.values()) and r['enrollment_qualified'] is True and r['accuracy_gate'] is None and r['counts']=={'enrolled_groups':152,'selected_substeps':608,'new_model_Torch_MJB_backend_physics':0} and r['cohort']=={'baseline_groups':24,'direction_groups':128,'donor_groups':0,'fresh_groups':0} and r['physical_force_or_gradient_qualified'] is r['whole_hidden_state_qualified'] is False)
    for v in [*sources.values(),*i.values(),*outputs.values()]:checked(v)
    gate('reviewer_no_Torch_model_MJB_force_decode_or_physics',not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor)
    np.savez_compressed(out/'independent_all152_target_arithmetic.npz',**{k:np.asarray(v) for k,v in values.items()})
    detail.update(source_files=len(sources),all_contact_indices_in_selected608=sum(len(s) for s in seen.values()),maximum_F64_implementation_difference=diff,
      inherited_scope='Conditional static metadata and decoded source vectors inherited; no new metadata/model/physics observation or wrench recomputation.',frame_scope='Raw fixed Reader palm frame distributed to4PRE per group, not608 independent mocap measurements. No inverse/project/torque-origin shift.',
      solver_scope='Primary solution independently computed by partial-pivot scalar elimination. Producer condition-number is retained descriptive SVD diagnostic, not independently requalified or a gate.',comparison_scope={'rtol':U.F64_RTOL,'atol':U.F64_ATOL,'not_a_defect_zero_threshold':True})
    result={'schema':'full151105_boundary_impulse_dataset_saved_independent:1','status':'PASS_SAVED_ALL152_ALL608_CONDITIONAL_TARGET_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_TARGET_DATASET_REVIEW','engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'independent_math':pin(MATH),'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'input_pins':i,'details':detail,'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'human_intent_qualified':False,'new_models_FW_AD_MJB_FK_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();metadata(a.protocol)
    if not a.run:print(json.dumps({'status':'PASS_METADATA_ONLY_NO_TARGET_VALUES','auditor':pin(__file__),'independent_math':pin(MATH)}));return 0
    require(a.run_dir is not None and a.output is not None,'Actual run/fresh output required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.output.resolve())
    except Exception as e:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'producer_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
